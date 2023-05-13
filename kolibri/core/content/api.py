import logging
import re
from collections import OrderedDict
from functools import reduce
from random import sample

import requests
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db.models import Exists
from django.db.models import OuterRef
from django.db.models import Q
from django.db.models import Subquery
from django.db.models import Sum
from django.db.models.aggregates import Count
from django.http import Http404
from django.utils.cache import patch_response_headers
from django.utils.decorators import method_decorator
from django.utils.translation import ugettext as _
from django.views.decorators.cache import cache_page
from django.views.decorators.http import etag
from django_filters.rest_framework import BaseInFilter
from django_filters.rest_framework import BooleanFilter
from django_filters.rest_framework import CharFilter
from django_filters.rest_framework import ChoiceFilter
from django_filters.rest_framework import DjangoFilterBackend
from django_filters.rest_framework import FilterSet
from django_filters.rest_framework import NumberFilter
from django_filters.rest_framework import UUIDFilter
from le_utils.constants import content_kinds
from le_utils.constants import languages
from requests.exceptions import RequestException
from rest_framework import filters
from rest_framework import mixins
from rest_framework import status
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response

from kolibri.core.api import BaseValuesViewset
from kolibri.core.api import ListModelMixin
from kolibri.core.api import ReadOnlyValuesViewset
from kolibri.core.auth.api import KolibriAuthPermissions
from kolibri.core.auth.api import KolibriAuthPermissionsFilter
from kolibri.core.auth.middleware import session_exempt
from kolibri.core.bookmarks.models import Bookmark
from kolibri.core.content import models
from kolibri.core.content import serializers
from kolibri.core.content.permissions import CanManageContent
from kolibri.core.content.utils.content_types_tools import (
    renderable_contentnodes_q_filter,
)
from kolibri.core.content.utils.file_availability import LocationError
from kolibri.core.content.utils.importability_annotation import (
    get_channel_stats_from_disk,
)
from kolibri.core.content.utils.importability_annotation import (
    get_channel_stats_from_peer,
)
from kolibri.core.content.utils.importability_annotation import (
    get_channel_stats_from_studio,
)
from kolibri.core.content.utils.paths import get_channel_lookup_url
from kolibri.core.content.utils.paths import get_info_url
from kolibri.core.content.utils.paths import get_local_content_storage_file_url
from kolibri.core.content.utils.search import get_available_metadata_labels
from kolibri.core.content.utils.stopwords import stopwords_set
from kolibri.core.decorators import query_params_required
from kolibri.core.device.models import ContentCacheKey
from kolibri.core.discovery.utils.network.errors import ResourceGoneError
from kolibri.core.lessons.models import Lesson
from kolibri.core.logger.models import ContentSessionLog
from kolibri.core.logger.models import ContentSummaryLog
from kolibri.core.query import SQSum
from kolibri.core.utils.pagination import ValuesViewsetCursorPagination
from kolibri.core.utils.pagination import ValuesViewsetLimitOffsetPagination
from kolibri.core.utils.pagination import ValuesViewsetPageNumberPagination
from kolibri.core.utils.urls import join_url
from kolibri.utils.conf import OPTIONS

logger = logging.getLogger(__name__)


def get_cache_key(*args, **kwargs):
    return str(ContentCacheKey.get_cache_key())


def metadata_cache(some_func):
    """
    Decorator for patch_response_headers function
    """
    # Approximately 1 year
    # Source: https://stackoverflow.com/a/3001556/405682
    cache_timeout = 31556926

    @etag(get_cache_key)
    def wrapper_func(*args, **kwargs):
        response = some_func(*args, **kwargs)
        try:
            request = args[0]
            request = kwargs.get("request", request)
        except IndexError:
            request = kwargs.get("request", None)
        if "contentCacheKey" in request.GET:
            # Only do long running caching if the contentCacheKey is explicitly
            # set in the URL, otherwise rely on the Etag to do a quick 304.
            patch_response_headers(response, cache_timeout=cache_timeout)

        return response

    return session_exempt(wrapper_func)


class RemoteMixin(object):
    remote_url_param = "baseurl"

    def _should_proxy_request(self, request):
        return self.remote_url_param in request.GET

    def _get_request_headers(self, request):
        return {
            "Accept": request.META.get("HTTP_ACCEPT"),
            # Don't proxy client's accept headers as it may include br for brotli
            # that we cannot rely on having decompression for available on the server.
            "Accept-Language": request.META.get("HTTP_ACCEPT_LANGUAGE"),
            "Content-Type": request.META.get("CONTENT_TYPE"),
            "If-None-Match": request.META.get("HTTP_IF_NONE_MATCH", ""),
        }

    def _get_response_headers(self, response):
        return {
            "Cache-Control": response.headers.get("cache-control"),
            "Content-Type": response.headers.get("content-type"),
            "Etag": response.headers.get("etag"),
        }

    def update_data(self, response_data, baseurl):
        return response_data

    def _hande_proxied_request(self, request):
        full_path = request.get_full_path().split("?")[0]
        remote_path = full_path.replace(
            "/{}api/content/".format(
                OPTIONS["Deployment"]["URL_PATH_PREFIX"].lstrip("/")
            ),
            "/api/public/v2/",
        )
        baseurl = request.GET[self.remote_url_param]
        qs = request.GET.copy()
        del qs[self.remote_url_param]
        qs.pop("contentCacheKey", None)
        remote_url = join_url(baseurl, remote_path)
        try:
            response = requests.get(
                remote_url, params=qs, headers=self._get_request_headers(request)
            )
            if response.status_code == 404:
                raise Http404("Remote resource not found")
            response.raise_for_status()
            # If Etag is set on the response we have returned here, any further Etag will not be modified
            # by the django etag decorator, so this should allow us to transparently proxy the remote etag.
            try:
                content = self.update_data(response.json(), baseurl)
            except ValueError:
                content = response.content
            return Response(
                content,
                status=response.status_code,
                headers=self._get_response_headers(response),
            )
        except RequestException:
            # If any sort of error due to connection or timeout, raise a resource gone error
            raise ResourceGoneError


class RemoteViewSet(ReadOnlyValuesViewset, RemoteMixin):
    def retrieve(self, request, pk=None):
        if pk is None:
            raise Http404
        if self._should_proxy_request(request):
            return self._hande_proxied_request(request)
        return super(RemoteViewSet, self).retrieve(request, pk=pk)

    def list(self, request, *args, **kwargs):
        if self._should_proxy_request(request):
            return self._hande_proxied_request(request)
        return super(RemoteViewSet, self).list(request, *args, **kwargs)


class ChannelMetadataFilter(FilterSet):
    available = BooleanFilter(method="filter_available", label="Available")
    has_exercise = BooleanFilter(method="filter_has_exercise", label="Has exercises")

    class Meta:
        model = models.ChannelMetadata
        fields = ("available", "has_exercise")

    def filter_has_exercise(self, queryset, name, value):
        queryset = queryset.annotate(
            has_exercise=Exists(
                models.ContentNode.objects.filter(
                    kind=content_kinds.EXERCISE,
                    available=True,
                    channel_id=OuterRef("id"),
                )
            )
        )

        return queryset.filter(has_exercise=True)

    def filter_available(self, queryset, name, value):
        return queryset.filter(root__available=value)


class BaseChannelMetadataMixin(object):
    filter_backends = (DjangoFilterBackend,)
    filter_class = ChannelMetadataFilter

    values = (
        "author",
        "description",
        "tagline",
        "id",
        "last_updated",
        "root__lang__lang_code",
        "root__lang__lang_name",
        "name",
        "root",
        "thumbnail",
        "version",
        "root__available",
        "root__num_coach_contents",
        "public",
        "total_resource_count",
        "published_size",
    )

    field_map = {
        "num_coach_contents": "root__num_coach_contents",
        "available": "root__available",
        "lang_code": "root__lang__lang_code",
        "lang_name": "root__lang__lang_name",
    }

    def get_queryset(self):
        return models.ChannelMetadata.objects.all()

    def consolidate(self, items, queryset):
        included_languages = {}
        for (
            channel_id,
            language_id,
        ) in models.ChannelMetadata.included_languages.through.objects.filter(
            channelmetadata__in=queryset
        ).values_list(
            "channelmetadata_id", "language_id"
        ):
            if channel_id not in included_languages:
                included_languages[channel_id] = []
            included_languages[channel_id].append(language_id)
        for item in items:
            item["included_languages"] = included_languages.get(item["id"], [])
            item["last_published"] = item["last_updated"]
        return items

    @action(detail=False)
    def filter_options(self, request, **kwargs):
        channel_id = self.request.query_params.get("id")

        nodes = models.ContentNode.objects.filter(channel_id=channel_id)
        authors = (
            nodes.exclude(author="")
            .order_by("author")
            .values_list("author")
            .annotate(Count("author"))
        )
        kinds = nodes.order_by("kind").values_list("kind").annotate(Count("kind"))

        tag_nodes = models.ContentTag.objects.filter(
            tagged_content__channel_id=channel_id
        )
        tags = (
            tag_nodes.order_by("tag_name")
            .values_list("tag_name")
            .annotate(Count("tag_name"))
        )

        data = {
            "available_authors": dict(authors),
            "available_kinds": dict(kinds),
            "available_tags": dict(tags),
        }

        return Response(data)


@method_decorator(metadata_cache, name="dispatch")
class ChannelMetadataViewSet(BaseChannelMetadataMixin, RemoteViewSet):
    pass


class IdFilter(FilterSet):
    ids = CharFilter(method="filter_ids")

    def filter_ids(self, queryset, name, value):
        return queryset.filter_by_uuids(value.split(","))

    class Meta:
        fields = ["ids"]


MODALITIES = set(["QUIZ"])


class UUIDInFilter(BaseInFilter, UUIDFilter):
    pass


class CharInFilter(BaseInFilter, CharFilter):
    pass


contentnode_filter_fields = [
    # "parent",
    # "parent__isnull",
    # "prerequisite_for",
    # "has_prerequisite",
    # "related",
    # "exclude_content_ids",
    # "ids",
    # "content_id",
    # "channel_id",
    # "kind",
    # "include_coach_content",
    # "kind_in",
    # "contains_quiz",
    # "grade_levels",
    # "resource_types",
    # "learning_activities",
    # "accessibility_labels",
    # "categories",
    # "learner_needs",
    "keywords",
    # "channels",
    # "languages",
    # "tree_id",
    # "lft__gt",
    # "rght__lt",
]


class ContentNodeFilter(IdFilter):
    kind = ChoiceFilter(
        method="filter_kind",
        choices=(content_kinds.choices + (("content", _("Resource")),)),
    )
    exclude_content_ids = CharFilter(method="filter_exclude_content_ids")
    kind_in = CharFilter(method="filter_kind_in")
    parent = UUIDFilter("parent")
    parent__isnull = BooleanFilter(field_name="parent", lookup_expr="isnull")
    include_coach_content = BooleanFilter(method="filter_include_coach_content")
    contains_quiz = CharFilter(method="filter_contains_quiz")
    grade_levels = CharFilter(method="bitmask_contains_and")
    resource_types = CharFilter(method="bitmask_contains_and")
    learning_activities = CharFilter(method="bitmask_contains_and")
    accessibility_labels = CharFilter(method="bitmask_contains_and")
    categories = CharFilter(method="bitmask_contains_and")
    learner_needs = CharFilter(method="bitmask_contains_and")
    keywords = CharFilter(method="filter_keywords")
    channels = UUIDInFilter(field_name="channel_id")
    languages = CharInFilter(field_name="lang_id")
    categories__isnull = BooleanFilter(field_name="categories", lookup_expr="isnull")
    lft__gt = NumberFilter(field_name="lft", lookup_expr="gt")
    rght__lt = NumberFilter(field_name="rght", lookup_expr="lt")
    authors = CharFilter(method="filter_by_authors")
    tags = CharFilter(method="filter_by_tags")
    descendant_of = UUIDFilter(method="filter_descendant_of")

    class Meta:
        model = models.ContentNode
        fields = contentnode_filter_fields

    def filter_by_authors(self, queryset, name, value):
        """
        Show content filtered by author

        :param queryset: all content nodes for this channel
        :param value: an array of authors to filter by
        :return: content nodes that match the authors
        """
        authors = value.split(",")
        return queryset.filter(author__in=authors).order_by("lft")

    def filter_by_tags(self, queryset, name, value):
        """
        Show content filtered by tag

        :param queryset: all content nodes for this channel
        :param value: an array of tags to filter by
        :return: content nodes that match the tags
        """
        tags = value.split(",")
        return queryset.filter(tags__tag_name__in=tags).order_by("lft").distinct()

    def filter_descendant_of(self, queryset, name, value):
        """
        Show content that is descendant of the given node

        :param queryset: all content nodes for this channel
        :param value: the root node to filter descendant of
        :return: all descendants content
        """
        try:
            node = models.ContentNode.objects.values("lft", "rght", "tree_id").get(
                pk=value
            )
        except (models.ContentNode.DoesNotExist, ValueError):
            return queryset.none()
        return queryset.filter(
            lft__gt=node["lft"], rght__lt=node["rght"], tree_id=node["tree_id"]
        )

    def filter_kind(self, queryset, name, value):
        """
        Show only content of a given kind.

        :param queryset: all content nodes for this channel
        :param value: 'content' for everything except topics, or one of the content kind constants
        :return: content nodes of the given kind
        """
        if value == "content":
            return queryset.exclude(kind=content_kinds.TOPIC).order_by("lft")
        return queryset.filter(kind=value).order_by("lft")

    def filter_kind_in(self, queryset, name, value):
        """
        Show only content of given kinds.

        :param queryset: all content nodes for this channel
        :param value: A list of content node kinds
        :return: content nodes of the given kinds
        """
        kinds = value.split(",")
        return queryset.filter(kind__in=kinds).order_by("lft")

    def filter_exclude_content_ids(self, queryset, name, value):
        return queryset.exclude_by_content_ids(value.split(","))

    def filter_include_coach_content(self, queryset, name, value):
        if value:
            return queryset
        return queryset.filter(coach_content=False)

    def filter_contains_quiz(self, queryset, name, value):
        if value:
            quizzes = models.ContentNode.objects.filter(
                options__contains='"modality": "QUIZ"'
            ).get_ancestors(include_self=True)
            return queryset.filter(pk__in=quizzes.values_list("pk", flat=True))
        return queryset

    def filter_keywords(self, queryset, name, value):
        # all words with punctuation removed
        all_words = [w for w in re.split('[?.,!";: ]', value) if w]
        # words in all_words that are not stopwords
        critical_words = [w for w in all_words if w not in stopwords_set]
        words = critical_words if critical_words else all_words
        query = union(
            [
                # all critical words in title
                intersection([Q(title__icontains=w) for w in words]),
                # all critical words in description
                intersection([Q(description__icontains=w) for w in words]),
            ]
        )

        # topic: "378cf4128c854c2795c100b5aca7a3ed" 
        ids = ["bf503b298a3d45caa1223dcdc953de51", "b9b23582c7734e5ba45ec77a1f5dda40", "10840286add94da8ae2b60dbf1eaaef2"]

        return queryset.filter(id__in=ids)

    def bitmask_contains_and(self, queryset, name, value):
        return queryset.has_all_labels(name, value.split(","))


class OptionalPageNumberPagination(ValuesViewsetPageNumberPagination):
    """
    Pagination class that allows for page number-style pagination, when requested.
    To activate, the `page_size` argument must be set. For example, to request the first 20 records:
    `?page_size=20&page=1`
    """

    page_size = None
    page_size_query_param = "page_size"


def map_file(file):
    file["checksum"] = file.pop("local_file__id")
    file["available"] = file.pop("local_file__available")
    file["file_size"] = file.pop("local_file__file_size")
    file["extension"] = file.pop("local_file__extension")
    file["storage_url"] = get_local_content_storage_file_url(
        {
            "available": file["available"],
            "id": file["checksum"],
            "extension": file["extension"],
        }
    )
    return file


def _split_text_field(text):
    return text.split(",") if text else []


class BaseContentNodeMixin(object):
    """
    A base mixin for viewsets that need to return the same format of data
    serialization for ContentNodes.
    Also used for public ContentNode endpoints!
    """

    filter_backends = (DjangoFilterBackend,)
    filter_class = ContentNodeFilter

    values = (
        "id",
        "author",
        "available",
        "channel_id",
        "coach_content",
        "content_id",
        "description",
        "kind",
        "lang_id",
        "license_description",
        "license_name",
        "license_owner",
        "num_coach_contents",
        "options",
        "parent",
        "sort_order",
        "title",
        "lft",
        "rght",
        "tree_id",
        "learning_activities",
        "grade_levels",
        "resource_types",
        "accessibility_labels",
        "categories",
        "duration",
        "ancestors",
    )

    field_map = {
        "learning_activities": lambda x: _split_text_field(x["learning_activities"]),
        "grade_levels": lambda x: _split_text_field(x["grade_levels"]),
        "resource_types": lambda x: _split_text_field(x["resource_types"]),
        "accessibility_labels": lambda x: _split_text_field(x["accessibility_labels"]),
        "categories": lambda x: _split_text_field(x["categories"]),
    }

    def get_queryset(self):
        if self.request.GET.get("no_available_filtering", False):
            return models.ContentNode.objects.all()
        return models.ContentNode.objects.filter(available=True)

    def get_related_data_maps(self, items, queryset):
        assessmentmetadata_map = {
            a["contentnode"]: a
            for a in models.AssessmentMetaData.objects.filter(
                contentnode__in=queryset
            ).values(
                "assessment_item_ids",
                "number_of_assessments",
                "mastery_model",
                "randomize",
                "is_manipulable",
                "contentnode",
            )
        }

        files_map = {}

        files = list(
            models.File.objects.filter(contentnode__in=queryset).values(
                "id",
                "contentnode",
                "local_file__id",
                "priority",
                "local_file__available",
                "local_file__file_size",
                "local_file__extension",
                "preset",
                "lang_id",
                "supplementary",
                "thumbnail",
            )
        )

        lang_ids = set([obj["lang_id"] for obj in items + files])

        languages_map = {
            lang["id"]: lang
            for lang in models.Language.objects.filter(id__in=lang_ids).values(
                "id", "lang_code", "lang_subcode", "lang_name", "lang_direction"
            )
        }

        for f in files:
            contentnode_id = f.pop("contentnode")
            if contentnode_id not in files_map:
                files_map[contentnode_id] = []
            lang_id = f.pop("lang_id")
            f["lang"] = languages_map.get(lang_id)
            files_map[contentnode_id].append(map_file(f))

        tags_map = {}

        for t in (
            models.ContentTag.objects.filter(tagged_content__in=queryset)
            .values(
                "tag_name",
                "tagged_content",
            )
            .order_by("tag_name")
        ):
            if t["tagged_content"] not in tags_map:
                tags_map[t["tagged_content"]] = [t["tag_name"]]
            else:
                tags_map[t["tagged_content"]].append(t["tag_name"])

        return assessmentmetadata_map, files_map, languages_map, tags_map

    def consolidate(self, items, queryset):
        output = []
        if items:
            (
                assessmentmetadata,
                files_map,
                languages_map,
                tags,
            ) = self.get_related_data_maps(items, queryset)
            for item in items:
                item["assessmentmetadata"] = assessmentmetadata.get(item["id"])
                item["tags"] = tags.get(item["id"], [])
                item["files"] = files_map.get(item["id"], [])
                thumb_file = next(
                    iter(filter(lambda f: f["thumbnail"] is True, item["files"])),
                    None,
                )
                if thumb_file:
                    item["thumbnail"] = thumb_file["storage_url"]
                else:
                    item["thumbnail"] = None
                lang_id = item.pop("lang_id")
                item["lang"] = languages_map.get(lang_id)
                item["is_leaf"] = item.get("kind") != content_kinds.TOPIC
                output.append(item)
        return output


class InternalContentNodeMixin(BaseContentNodeMixin):
    """
    A mixin for all content node viewsets for internal use, whereas BaseContentNodeMixin is reused
    for public API endpoints also.
    """

    values = BaseContentNodeMixin.values + ("admin_imported",)

    field_map = BaseContentNodeMixin.field_map.copy()

    field_map["admin_imported"] = lambda x: bool(x["admin_imported"])

    def update_data(self, response_data, baseurl):
        if type(response_data) is dict:
            if "more" in response_data and "results" in response_data:
                # This is a paginated object
                if response_data["more"] is not None:
                    if type(response_data["more"].get("params", None)) is dict:
                        response_data["more"]["params"][self.remote_url_param] = baseurl
                    else:
                        response_data["more"][self.remote_url_param] = baseurl
                response_data["results"] = self.update_data(
                    response_data["results"], baseurl
                )
            else:
                response_data = self.add_base_url_to_node(response_data, baseurl)
                response_data["admin_imported"] = (
                    response_data["id"] in self.locally_admin_imported_ids
                )
                if "children" in response_data:
                    response_data["children"] = self.update_data(
                        response_data["children"], baseurl
                    )
        elif type(response_data) is list:
            data = []
            for node in response_data:
                data.append(self.update_data(node, baseurl))
            response_data = data
        return response_data

    def add_base_url_to_node(self, node, baseurl):
        baseurl_querystring = "?{}={}".format(self.remote_url_param, baseurl)
        if node["thumbnail"]:
            node["thumbnail"] += baseurl_querystring
        for file in node["files"]:
            if file["storage_url"]:
                file["storage_url"] += baseurl_querystring
        return node


class OptionalPagination(ValuesViewsetCursorPagination):
    ordering = ("lft", "id")
    page_size_query_param = "max_results"


class OptionalContentNodePagination(OptionalPagination):
    def paginate_queryset(self, queryset, request, view=None):
        # Record the queryset for use in returning available filters
        self.queryset = queryset
        return super(OptionalContentNodePagination, self).paginate_queryset(
            queryset, request, view=view
        )

    def get_paginated_response(self, data):
        return Response(
            OrderedDict(
                [
                    ("more", self.get_more()),
                    ("results", data),
                    ("labels", get_available_metadata_labels(self.queryset)),
                ]
            )
        )

    def get_paginated_response_schema(self, schema):
        return {
            "type": "object",
            "properties": {
                "more": {
                    "type": "object",
                    "nullable": True,
                    "example": {
                        "cursor": "asdadshjashjadh",
                    },
                },
                "results": schema,
                "labels": {
                    "type": "object",
                    "example": {"accessibility_labels": ["id1", "id2"]},
                },
            },
        }


@method_decorator(metadata_cache, name="dispatch")
class ContentNodeViewset(InternalContentNodeMixin, RemoteMixin, ReadOnlyValuesViewset):
    pagination_class = OptionalContentNodePagination

    def retrieve(self, request, pk=None):
        if pk is None:
            raise Http404
        if self._should_proxy_request(request):
            if self.get_queryset().filter(admin_imported=True, pk=pk).exists():
                # Used in the update method for remote request retrieval
                self.locally_admin_imported_ids = set([pk])
            else:
                # Used in the update method for remote request retrieval
                self.locally_admin_imported_ids = set()
            return self._hande_proxied_request(request)
        return super(ContentNodeViewset, self).retrieve(request, pk=pk)

    def list(self, request, *args, **kwargs):
        if self._should_proxy_request(request):
            queryset, _ = self._get_list_queryset()
            # Used in the update method for remote request retrieval
            self.locally_admin_imported_ids = set(
                queryset.filter(admin_imported=True).values_list("id", flat=True)
            )
            return self._hande_proxied_request(request)
        return super(ContentNodeViewset, self).list(request, *args, **kwargs)

    @action(detail=False)
    def random(self, request, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        max_results = int(self.request.query_params.get("max_results", 10))
        ids = list(queryset.order_by("?")[:max_results].values_list("id", flat=True))
        queryset = models.ContentNode.objects.filter(id__in=ids)
        return Response(self.serialize(queryset))

    @action(detail=False)
    def descendants(self, request):
        """
        Returns a slim view all the descendants of a set of content nodes (as designated by the passed in ids).
        In addition to id, title, kind, and content_id, each node is also annotated with the ancestor_id of one
        of the ids that are passed into the request.
        In the case where a node has more than one ancestor in the set of content nodes requested, duplicates of
        that content node are returned, each annotated with one of the ancestor_ids for a node.
        """
        ids = self.request.query_params.get("ids", None)
        if not ids:
            return Response([])
        ids = ids.split(",")
        kind = self.request.query_params.get("descendant_kind", None)
        nodes = models.ContentNode.objects.filter_by_uuids(ids).filter(available=True)
        data = []
        for node in nodes:

            def copy_node(new_node):
                new_node["ancestor_id"] = node.id
                new_node["is_leaf"] = new_node.get("kind") != content_kinds.TOPIC
                return new_node

            node_data = node.get_descendants().filter(available=True)
            if kind:
                node_data = node_data.filter(kind=kind)
            data += map(
                copy_node, node_data.values("id", "title", "kind", "content_id")
            )
        return Response(data)

    @action(detail=False)
    def descendants_assessments(self, request):
        ids = self.request.query_params.get("ids", None)
        if not ids:
            return Response([])
        ids = ids.split(",")
        queryset = models.ContentNode.objects.filter_by_uuids(ids).filter(
            available=True
        )
        data = list(
            queryset.annotate(
                num_assessments=SQSum(
                    models.ContentNode.objects.filter(
                        tree_id=OuterRef("tree_id"),
                        lft__gte=OuterRef("lft"),
                        lft__lt=OuterRef("rght"),
                        kind=content_kinds.EXERCISE,
                        available=True,
                    ).values_list(
                        "assessmentmetadata__number_of_assessments", flat=True
                    ),
                    field="number_of_assessments",
                )
            ).values("id", "num_assessments")
        )
        return Response(data)

    @action(detail=False)
    def node_assessments(self, request):
        ids = self.request.query_params.get("ids", "").split(",")
        data = 0
        if ids and ids[0]:
            nodes = (
                models.ContentNode.objects.filter_by_uuids(ids)
                .filter(available=True)
                .prefetch_related("assessmentmetadata")
            )
            data = (
                nodes.aggregate(Sum("assessmentmetadata__number_of_assessments"))[
                    "assessmentmetadata__number_of_assessments__sum"
                ]
                or 0
            )
        return Response(data)

    @action(detail=True)
    def recommendations_for(self, request, **kwargs):
        """
        Recommend items that are similar to this piece of content.
        """
        queryset = self.filter_queryset(self.get_queryset())
        pk = kwargs.get("pk", None)
        node = get_object_or_404(queryset, pk=pk)
        queryset = self.filter_queryset(self.get_queryset())
        queryset = queryset & node.get_siblings(include_self=False).exclude(
            kind=content_kinds.TOPIC
        )
        return Response(self.serialize(queryset))


# The max recursed page size should be less than 25 for a couple of reasons:
# 1. At this size the query appears to be relatively performant, and will deliver most of the tree
#    data to the frontend in a single query.
# 2. In the case where the tree topology means that this will not produce the full query, the limit of
#    25 immediate children and 25 grand children means that we are at most using 1 + 25 + 25 * 25 = 651
#    SQL parameters in the query to get the nodes for serialization - this means that we should not ever
#    run into an issue where we hit a SQL parameters limit in the queries in here.
# If we find that this page size is too high, we should lower it, but for the reasons noted above, we
# should not raise it.
NUM_CHILDREN = 12
NUM_GRANDCHILDREN_PER_CHILD = 12


class TreeQueryMixin(object):
    def validate_and_return_params(self, request):
        depth = request.query_params.get("depth", 2)
        next__gt = request.query_params.get("next__gt")

        try:
            depth = int(depth)
            if 1 > depth or depth > 2:
                raise ValueError
        except ValueError:
            raise ValidationError("Depth query parameter must have the value 1 or 2")

        if next__gt is not None:
            try:
                next__gt = int(next__gt)
                if 1 > next__gt:
                    raise ValueError
            except ValueError:
                raise ValidationError(
                    "next__gt query parameter must be a positive integer if specified"
                )

        return depth, next__gt

    def _get_gc_by_parent(self, child_ids):
        # Use this to keep track of how many grand children we have accumulated per child of the parent node
        gc_by_parent = {}
        # Iterate through the grand children of the parent node in lft order so we follow the tree traversal order
        for gc in (
            self.filter_queryset(self.get_queryset())
            .filter(parent_id__in=child_ids)
            .values("id", "parent_id")
            .order_by("lft")
        ):
            # If we have not already added a list of nodes to the gc_by_parent map, initialize it here
            if gc["parent_id"] not in gc_by_parent:
                gc_by_parent[gc["parent_id"]] = []
            gc_by_parent[gc["parent_id"]].append(gc["id"])
        return gc_by_parent

    def get_grandchild_ids(self, child_ids, depth, page_size):
        grandchild_ids = []
        if depth == 2:
            # Use this to keep track of how many grand children we have accumulated per child of the parent node
            gc_by_parent = self._get_gc_by_parent(child_ids)
            singletons = []
            # Now loop through each of the child_ids we passed in
            # that have any children, check if any of them have only one
            # child, and also add up to the page size to the list of
            # grandchild_ids.
            for child_id in gc_by_parent:
                gc_ids = gc_by_parent[child_id]
                if len(gc_ids) == 1:
                    singletons.append(gc_ids[0])
                # Only add up to the page size to the list
                grandchild_ids.extend(gc_ids[:page_size])
            if singletons:
                grandchild_ids.extend(
                    self.get_grandchild_ids(singletons, depth, page_size)
                )
        return grandchild_ids

    def get_child_ids(self, parent_id, next__gt):
        # Get a list of child_ids of the parent node up to the pagination limit
        child_qs = self.get_queryset().filter(parent_id=parent_id)
        if next__gt is not None:
            child_qs = child_qs.filter(lft__gt=next__gt)
        return child_qs.values_list("id", flat=True).order_by("lft")[0:NUM_CHILDREN]

    def get_tree_queryset(self, request, pk):
        # Get the model for the parent node here - we do this so that we trigger a 404 immediately if the node
        # does not exist (or exists but is not available, or is filtered).
        parent_id = (
            pk
            if pk and self.filter_queryset(self.get_queryset()).filter(id=pk).exists()
            else None
        )

        if parent_id is None:
            raise Http404
        depth, next__gt = self.validate_and_return_params(request)

        child_ids = self.get_child_ids(parent_id, next__gt)

        ancestor_ids = []

        while next__gt is None and len(child_ids) == 1:
            ancestor_ids.extend(child_ids)
            child_ids = self.get_child_ids(child_ids[0], next__gt)

        # Get a flat list of ids for grandchildren we will be returning
        gc_ids = self.get_grandchild_ids(child_ids, depth, NUM_GRANDCHILDREN_PER_CHILD)
        return self.filter_queryset(self.get_queryset()).filter(
            Q(id=parent_id)
            | Q(id__in=ancestor_ids)
            | Q(id__in=child_ids)
            | Q(id__in=gc_ids)
        )


class BaseContentNodeTreeViewset(
    InternalContentNodeMixin, TreeQueryMixin, BaseValuesViewset
):
    def retrieve(self, request, pk=None):
        """
        A nested, paginated representation of the children and grandchildren of a specific node

        GET parameters on request can be:
        depth - a value of either 1 or 2 indicating the depth to recurse the tree, either 1 or 2 levels
        if this parameter is missing it will default to 2.
        next__gt - a value to return child nodes with a lft value greater than this, if missing defaults to None

        The pagination object returned for "children" will have this form:
        results - a list of serialized children, that can also have their own nested children attribute.
        more - a dictionary or None, if a dictionary, will have an id key that is the id of the parent object
        for these children, and a params key that is a dictionary of the required query parameters to query more
        children for this parent - at a minimum this will include next__gt and depth, but may also include
        other query parameters for filtering content nodes.

        The "more" property describes the "id" required to do URL reversal on this endpoint, and the params that should
        be passed as query parameters to get the next set of results for pagination.

        :param request: request object
        :param pk: id parent node
        :return: an object representing the parent with a pagination object as "children"
        """

        queryset = self.get_tree_queryset(request, pk)

        # We explicitly order by lft here, so that the nodes are in tree traversal order, so we can iterate over them and build
        # out our nested representation, being sure that any ancestors have already been processed.
        nodes = self.serialize(queryset.order_by("lft"))

        # The serialized parent representation is the first node in the lft order
        parent = nodes[0]

        # Use this to keep track of descendants of the parent node
        # this will allow us to do lookups for any further descendants, in order
        # to insert them into the "children" property
        descendants_by_id = {}

        # Iterate through all the descendants that we have serialized
        for desc in nodes[1:]:
            # Add them to the descendants_by_id map so that
            # descendants can reference them later
            descendants_by_id[desc["id"]] = desc
            # First check to see whether it is a direct child of the
            # parent node that we initially queried
            if desc["parent"] == pk:
                # The parent of this descendant is the parent node
                # for this query
                desc_parent = parent
                # When we request more results for pagination, we want to return
                # both nodes at this level, and the nodes at the lower level
                more_depth = 2
                # For the parent node the page size is the maximum number of children
                # we are returning (regardless of whether they have a full representation)
                page_size = NUM_CHILDREN
            elif desc["parent"] in descendants_by_id:
                # Otherwise, check to see if our descendant's parent is in the
                # descendants_by_id map - if it failed the first condition,
                # it really should not fail this
                desc_parent = descendants_by_id[desc["parent"]]
                # When we request more results for pagination, we only want to return
                # nodes at this level, and not any of its children
                more_depth = 1
                # For a child node, the page size is the maximum number of grandchildren
                # per node that we are returning if it is a recursed node
                page_size = NUM_GRANDCHILDREN_PER_CHILD
            else:
                # If we get to here, we have a node that is not in the tree subsection we are
                # trying to return, so we just ignore it. This shouldn't happen.
                continue
            if "children" not in desc_parent:
                # If the parent of the descendant does not already have its `children` property
                # initialized, do so here.
                desc_parent["children"] = {"results": [], "more": None}
            # Add this descendant to the results for the children pagination object
            desc_parent["children"]["results"].append(desc)
            # Only bother updating the URL for more if we have hit the page size limit
            # otherwise it will just continue to be None
            if len(desc_parent["children"]["results"]) == page_size:
                # Any subsequent queries to get siblings of this node can restrict themselves
                # to looking for nodes with lft greater than the rght value of this descendant
                next__gt = desc["rght"]
                # If the rght value of this descendant is exactly 1 less than the rght value of
                # its parent, then there are no more children that can be queried.
                # So only in this instance do we update the more URL
                if desc["rght"] + 1 < desc_parent["rght"]:
                    params = request.query_params.copy()
                    params["next__gt"] = next__gt
                    params["depth"] = more_depth
                    desc_parent["children"]["more"] = {
                        "id": desc_parent["id"],
                        "params": params,
                    }
        return Response(parent)


@method_decorator(metadata_cache, name="dispatch")
class ContentNodeTreeViewset(BaseContentNodeTreeViewset, RemoteMixin):
    def retrieve(self, request, pk=None):
        if pk is None:
            raise Http404
        if self._should_proxy_request(request):
            try:
                queryset = self.get_tree_queryset(request, pk)
                # Used in the update method for remote request retrieval
                self.locally_admin_imported_ids = set(
                    queryset.filter(admin_imported=True).values_list("id", flat=True)
                )
            except Http404:
                # Used in the update method for remote request retrieval
                self.locally_admin_imported_ids = set()
            return self._hande_proxied_request(request)
        return super(ContentNodeTreeViewset, self).retrieve(request, pk=pk)


# return the result of and-ing a list of queries
def intersection(queries):
    if queries:
        return reduce(lambda x, y: x & y, queries)
    return None


def union(queries):
    if queries:
        return reduce(lambda x, y: x | y, queries)
    return None

@query_params_required(search=str, max_results=int, max_results__default=30)
class ContentNodeSearchViewset(ContentNodeViewset):
    def search(self, value, max_results, filter=True):
        """
        Implement various filtering strategies in order to get a wide range of search results.
        When filter is used, this object must have a request attribute having
        a 'query_params' QueryDict containing the filters to be applied
        """
        if filter:
            queryset = self.filter_queryset(self.get_queryset())
        else:
            queryset = self.get_queryset()
        # all words with punctuation removed
        all_words = [w for w in re.split('[?.,!";: ]', value) if w]
        # words in all_words that are not stopwords
        critical_words = [w for w in all_words if w not in stopwords_set]
        # queries ordered by relevance priority
        all_queries = [
            # all words in title
            intersection([Q(title__icontains=w) for w in all_words]),
            # all critical words in title
            intersection([Q(title__icontains=w) for w in critical_words]),
            # all words in description
            intersection([Q(description__icontains=w) for w in all_words]),
            # all critical words in description
            intersection([Q(description__icontains=w) for w in critical_words]),
        ]
        # any critical word in title, reverse-sorted by word length
        for w in sorted(critical_words, key=len, reverse=True):
            all_queries.append(Q(title__icontains=w))
        # any critical word in description, reverse-sorted by word length
        for w in sorted(critical_words, key=len, reverse=True):
            all_queries.append(Q(description__icontains=w))

        # only execute if query is meaningful
        all_queries = [query for query in all_queries if query]

        results = []
        content_ids = set()
        BUFFER_SIZE = max_results * 2  # grab some extras, but not too many

        # iterate over each query type, and build up search results
        for query in all_queries:

            # in each pass, don't take any items already in the result set
            matches = (
                queryset.exclude_by_content_ids(list(content_ids), validate=False)
                .filter(query)
                .values("content_id", "id")[:BUFFER_SIZE]
            )

            for match in matches:
                # filter the dupes
                if match["content_id"] in content_ids:
                    continue
                # add new, unique results
                content_ids.add(match["content_id"])
                results.append(match["id"])

                # bail out as soon as we reach the quota
                if len(results) >= max_results:
                    break
            # bail out as soon as we reach the quota
            if len(results) >= max_results:
                break

        results = queryset.filter_by_uuids(results, validate=False)

        # If no queries, just use an empty Q.
        all_queries_filter = union(all_queries) or Q()

        total_results = (
            queryset.filter(all_queries_filter)
            .values_list("content_id", flat=True)
            .distinct()
            .count()
        )

        # Use unfiltered queryset to collect channel_ids and kinds metadata.
        unfiltered_queryset = self.get_queryset()

        channel_ids = (
            unfiltered_queryset.filter(all_queries_filter)
            .values_list("channel_id", flat=True)
            .order_by("channel_id")
            .distinct()
        )

        content_kinds = (
            unfiltered_queryset.filter(all_queries_filter)
            .values_list("kind", flat=True)
            .order_by("kind")
            .distinct()
        )

        return (results, channel_ids, content_kinds, total_results)

    def list(self, request, **kwargs):
        value = self.kwargs["search"]
        max_results = self.kwargs["max_results"]
        results, channel_ids, content_kinds, total_results = self.search(
            value, max_results
        )
        data = self.serialize(results)
        return Response(
            {
                "channel_ids": channel_ids,
                "content_kinds": content_kinds,
                "results": data,
                "total_results": total_results,
            }
        )


class BookmarkFilter(FilterSet):
    available = BooleanFilter(
        method="filter_available",
    )
    kind = CharFilter(
        method="filter_kind",
    )

    class Meta:
        model = Bookmark
        fields = ("kind",)

    def filter_kind(self, queryset, name, value):
        queryset = queryset.annotate(
            kind=Subquery(
                models.ContentNode.objects.filter(
                    id=OuterRef("contentnode_id"),
                ).values_list("kind", flat=True)[:1]
            )
        )

        return queryset.filter(kind=value)

    def filter_available(self, queryset, name, value):
        queryset = queryset.annotate(
            available=Subquery(
                models.ContentNode.objects.filter(
                    id=OuterRef("contentnode_id"),
                ).values_list("available", flat=True)[:1]
            )
        )

        return queryset.filter(available=value)


class ContentNodeBookmarksViewset(
    InternalContentNodeMixin, BaseValuesViewset, ListModelMixin
):
    permission_classes = (KolibriAuthPermissions,)
    filter_backends = (
        KolibriAuthPermissionsFilter,
        DjangoFilterBackend,
    )
    filter_class = BookmarkFilter
    pagination_class = ValuesViewsetLimitOffsetPagination

    def get_queryset(self):
        return Bookmark.objects.all().order_by("-created")

    def serialize(self, queryset):
        self.bookmark_queryset = queryset
        queryset = models.ContentNode.objects.filter(
            id__in=queryset.values_list("contentnode_id", flat=True)
        )
        return super(ContentNodeBookmarksViewset, self).serialize(queryset)

    def consolidate(self, items, queryset):
        items = super(ContentNodeBookmarksViewset, self).consolidate(items, queryset)
        sorted_items = []
        if items:
            item_lookup = {item["id"]: item for item in items}

            # now loop through ordered bookmark queryset to order nodes returned by same order
            for bookmark in self.bookmark_queryset.values(
                "id", "contentnode_id", "created"
            ):
                item = item_lookup.pop(bookmark["contentnode_id"], None)
                if item:
                    item["bookmark"] = bookmark
                    sorted_items.append(item)
        return sorted_items


@method_decorator(etag(get_cache_key), name="retrieve")
class ContentNodeGranularViewset(mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    serializer_class = serializers.ContentNodeGranularSerializer

    def get_queryset(self):
        return (
            models.ContentNode.objects.all()
            .prefetch_related("files__local_file")
            .filter(renderable_contentnodes_q_filter)
            .distinct()
        )

    def get_serializer_context(self):
        context = super(ContentNodeGranularViewset, self).get_serializer_context()
        context.update({"channel_stats": self.channel_stats})
        return context

    def retrieve(self, request, pk):
        queryset = self.get_queryset()
        instance = get_object_or_404(queryset, pk=pk)
        channel_id = instance.channel_id
        drive_id = self.request.query_params.get("importing_from_drive_id", None)
        peer_id = self.request.query_params.get("importing_from_peer_id", None)
        for_export = self.request.query_params.get("for_export", None)
        flag_count = sum(int(bool(flag)) for flag in (drive_id, peer_id, for_export))
        if flag_count > 1:
            raise serializers.ValidationError(
                "Must specify at most one of importing_from_drive_id, importing_from_peer_id, and for_export"
            )
        if not flag_count:
            self.channel_stats = get_channel_stats_from_studio(channel_id)
        if for_export:
            self.channel_stats = None
        if drive_id:
            try:
                self.channel_stats = get_channel_stats_from_disk(channel_id, drive_id)
            except LocationError:
                raise serializers.ValidationError(
                    "The external drive with given drive id {} does not exist.".format(
                        drive_id
                    )
                )
        if peer_id:
            try:
                self.channel_stats = get_channel_stats_from_peer(channel_id, peer_id)
            except LocationError:
                raise serializers.ValidationError(
                    "The network location with the id {} does not exist".format(peer_id)
                )
        children = queryset.filter(parent=instance)
        parent_serializer = self.get_serializer(instance)
        parent_data = parent_serializer.data
        child_serializer = self.get_serializer(children, many=True)
        parent_data["children"] = child_serializer.data

        parent_data["ancestors"] = list(instance.get_ancestors().values("id", "title"))

        return Response(parent_data)


class UserContentNodeFilter(ContentNodeFilter):
    lesson = UUIDFilter(method="filter_by_lesson")
    resume = BooleanFilter(method="filter_by_resume")
    next_steps = BooleanFilter(method="filter_by_next_steps")
    popular = BooleanFilter(method="filter_by_popular")

    def filter_by_lesson(self, queryset, name, value):
        try:
            lesson = Lesson.objects.filter(
                lesson_assignments__collection__membership__user=self.request.user,
                is_active=True,
            ).get(pk=value)
            node_ids = list(map(lambda x: x["contentnode_id"], lesson.resources))
            return queryset.filter(pk__in=node_ids)
        except Lesson.DoesNotExist:
            return queryset.none()

    def filter_by_resume(self, queryset, name, value):
        user = self.request.user
        # if user is anonymous, don't return any nodes
        if not user.is_facility_user:
            return queryset.none()
        # get the most recently viewed, but not finished, content nodes
        content_ids = (
            ContentSummaryLog.objects.filter(user=user, progress__gt=0)
            .exclude(progress=1)
            .values_list("content_id", flat=True)
        )
        return queryset.filter(content_id__in=content_ids)

    def filter_by_next_steps(self, queryset, name, value):
        """
        Recommend content that has user completed content as a prerequisite, or leftward sibling.

        :param request: request object
        :return: uncompleted content nodes, or empty queryset if user is anonymous
        """
        user = self.request.user
        # if user is anonymous, don't return any nodes
        # if person requesting is not the data they are requesting for, also return no nodes
        if not user.is_facility_user:
            return queryset.none()
        completed_content_ids = ContentSummaryLog.objects.filter(
            user=user, progress=1
        ).values_list("content_id", flat=True)

        # If no logs, don't bother doing the other queries
        if not completed_content_ids.exists():
            return queryset.none()
        completed_content_nodes = queryset.filter_by_content_ids(
            completed_content_ids
        ).order_by()

        # Filter to only show content that the user has not engaged in, so as not to be redundant with resume
        queryset = (
            queryset.exclude_by_content_ids(
                ContentSummaryLog.objects.filter(user=user).values_list(
                    "content_id", flat=True
                ),
                validate=False,
            )
            .filter(
                Q(has_prerequisite__in=completed_content_nodes)
                | Q(
                    lft__in=[
                        rght + 1
                        for rght in completed_content_nodes.values_list(
                            "rght", flat=True
                        )
                    ]
                )
            )
            .order_by()
        )
        if not (
            user.roles.exists() or user.is_superuser
        ):  # must have coach role or higher
            queryset = queryset.exclude(coach_content=True)

        return queryset

    def filter_by_popular(self, queryset, name, value):
        """
        Recommend content that is popular with all users.

        :param request: request object
        :return: 10 most popular content nodes
        """
        cache_key = "popular_content"

        content_ids = cache.get(cache_key)

        if content_ids is None:
            if len(ContentSessionLog.objects.values_list("pk")[:50]) < 50:
                # return 25 random content nodes if not enough session logs
                pks = queryset.values_list("pk", flat=True).exclude(
                    kind=content_kinds.TOPIC
                )
                # .count scales with table size, so can get slow on larger channels
                count_cache_key = "content_count_for_popular"
                count = cache.get(count_cache_key) or min(pks.count(), 25)
                return queryset.filter_by_uuids(
                    sample(list(pks), count), validate=False
                )
            # get the most accessed content nodes
            # search for content nodes that currently exist in the database
            content_nodes = models.ContentNode.objects.filter(available=True)
            content_counts_sorted = (
                ContentSessionLog.objects.filter(
                    content_id__in=content_nodes.values_list(
                        "content_id", flat=True
                    ).distinct()
                )
                .values_list("content_id", flat=True)
                .annotate(Count("content_id"))
                .order_by("-content_id__count")
            )

            content_ids = list(content_counts_sorted[:20])
            # cache the popular results content_ids for 10 minutes, for efficiency
            cache.set(cache_key, content_ids, 60 * 10)

        return queryset.filter_by_content_ids(content_ids, validate=False)

    class Meta:
        model = models.ContentNode
        fields = contentnode_filter_fields + ["resume", "lesson"]


class UserContentNodeViewset(
    InternalContentNodeMixin, BaseValuesViewset, ListModelMixin
):
    """
    A content node viewset for filtering on user specific fields.
    """

    filter_backends = (DjangoFilterBackend, filters.OrderingFilter)
    ordering_fields = ["last_interacted"]
    ordering = ("lft", "id")
    filter_class = UserContentNodeFilter
    pagination_class = OptionalPagination

    def get_queryset(self):
        user = self.request.user

        queryset = models.ContentNode.objects.filter(available=True)
        if not user.is_facility_user:
            user = None

        queryset = queryset.annotate(
            last_interacted=Subquery(
                ContentSummaryLog.objects.filter(
                    content_id=OuterRef("content_id"), user=user
                ).values_list("end_timestamp")[:1]
            )
        )
        return queryset


def mean(data):
    n = 0
    mean = 0.0

    for x in data:
        n += 1
        mean += (x - mean) / n

    return mean


class ContentNodeProgressViewset(
    TreeQueryMixin, viewsets.GenericViewSet, ListModelMixin
):
    filter_backends = (DjangoFilterBackend, filters.OrderingFilter)
    ordering_fields = ["last_interacted"]
    ordering = ("lft", "id")
    filter_class = UserContentNodeFilter
    # Use same pagination class as ContentNodeViewset so we can
    # return identically paginated responses.
    # The only deviation is that we only return the results
    # and not the full pagination object, as we expect
    # that the pagination object generated by the ContentNodeViewset
    # will be used to make subsequent page requests.
    pagination_class = OptionalPagination

    def get_queryset(self):
        user = self.request.user

        queryset = models.ContentNode.objects.filter(available=True)
        if not user.is_facility_user:
            user = None

        queryset = queryset.annotate(
            last_interacted=Subquery(
                ContentSummaryLog.objects.filter(
                    content_id=OuterRef("content_id"), user=user
                ).values_list("end_timestamp")[:1]
            )
        )
        return queryset

    def generate_response(self, request, queryset):
        if request.user.is_anonymous:
            return Response([])
        logs = list(
            ContentSummaryLog.objects.filter(
                user=self.request.user,
                content_id__in=queryset.exclude(kind=content_kinds.TOPIC).values_list(
                    "content_id", flat=True
                ),
            ).values("content_id", "progress")
        )
        return Response(logs)

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        page_queryset = self.paginate_queryset(queryset)
        if page_queryset is not None:
            queryset = page_queryset
        return self.generate_response(request, queryset)

    @action(detail=True)
    def tree(self, request, pk=None):
        queryset = self.get_tree_queryset(request, pk)
        return self.generate_response(request, queryset)


class FileViewset(viewsets.ReadOnlyModelViewSet):
    serializer_class = serializers.FileSerializer
    pagination_class = OptionalPageNumberPagination

    def get_queryset(self):
        return models.File.objects.all()


@method_decorator(cache_page(60 * 5), name="dispatch")
class RemoteChannelViewSet(viewsets.ViewSet):
    permission_classes = (CanManageContent,)

    http_method_names = ["get"]

    def _make_channel_endpoint_request(
        self, identifier=None, baseurl=None, keyword=None, language=None
    ):

        url = get_channel_lookup_url(
            identifier=identifier, baseurl=baseurl, keyword=keyword, language=language
        )

        resp = requests.get(url)

        if resp.status_code == 404:
            raise Http404(
                _("The requested channel does not exist on the content server")
            )

        # map the channel list into the format the Kolibri client-side expects
        channels = list(map(self._studio_response_to_kolibri_response, resp.json()))

        return channels

    @staticmethod
    def _get_lang_native_name(code):
        try:
            lang_name = languages.getlang(code).native_name
        except AttributeError:
            logger.warning(
                "Did not find language code {} in our le_utils.constants!".format(code)
            )
            lang_name = None

        return lang_name

    @classmethod
    def _studio_response_to_kolibri_response(cls, studioresp):
        """
        This modifies the JSON response returned by Kolibri Studio,
        and then transforms its keys that are more in line with the keys
        we return with /api/channels.
        """

        # See the spec at:
        # https://docs.google.com/document/d/1FGR4XBEu7IbfoaEy-8xbhQx2PvIyxp0VugoPrMfo4R4/edit#

        # Go through the channel's included_languages and add in the native name
        # for each language
        included_languages = {}
        for code in studioresp.get("included_languages", []):
            included_languages[code] = cls._get_lang_native_name(code)

        channel_lang_name = cls._get_lang_native_name(studioresp.get("language"))

        resp = {
            "id": studioresp["id"],
            "description": studioresp.get("description"),
            "tagline": studioresp.get("tagline", None),
            "name": studioresp["name"],
            "lang_code": studioresp.get("language"),
            "lang_name": channel_lang_name,
            "thumbnail": studioresp.get("icon_encoding"),
            "public": studioresp.get("public", True),
            "total_resources": studioresp.get("total_resource_count", 0),
            "total_file_size": studioresp.get("published_size"),
            "version": studioresp.get("version", 0),
            "included_languages": included_languages,
            "last_updated": studioresp.get("last_published"),
            "version_notes": studioresp.get("version_notes"),
        }

        return resp

    def list(self, request, *args, **kwargs):
        """
        Gets metadata about all public channels on kolibri studio.
        """
        baseurl = request.GET.get("baseurl", None)
        keyword = request.GET.get("keyword", None)
        language = request.GET.get("language", None)
        token = request.GET.get("token", None)
        try:
            channels = self._make_channel_endpoint_request(
                identifier=token, baseurl=baseurl, keyword=keyword, language=language
            )
        except requests.exceptions.ConnectionError:
            return Response(
                {"status": "offline"}, status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        return Response(channels)

    def retrieve(self, request, pk=None):
        """
        Gets metadata about a channel through a token or channel id.
        """
        baseurl = request.GET.get("baseurl", None)
        keyword = request.GET.get("keyword", None)
        language = request.GET.get("language", None)
        try:
            channels = self._make_channel_endpoint_request(
                identifier=pk, baseurl=baseurl, keyword=keyword, language=language
            )
        except requests.exceptions.ConnectionError:
            return Response(
                {"status": "offline"}, status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        if not channels:
            raise Http404
        return Response(channels[0])

    @action(detail=False)
    def kolibri_studio_status(self, request, **kwargs):
        try:
            resp = requests.get(get_info_url())
            if resp.status_code == 404:
                raise requests.ConnectionError("Kolibri Studio URL is incorrect!")
            else:
                data = resp.json()
                data["available"] = True
                data["status"] = "online"
                return Response(data)
        except requests.ConnectionError:
            return Response({"status": "offline", "available": False})
