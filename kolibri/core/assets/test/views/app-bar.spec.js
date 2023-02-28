import { shallowMount, createLocalVue } from '@vue/test-utils';
import Vuex from 'vuex';
import AppBar from '../../src/views/AppBar';

jest.mock('kolibri.urls');

const localVue = createLocalVue();
localVue.use(Vuex);
const store = new Vuex.Store({
  state: {},
  getters: {
    isAppContext: () => false,
  },
});

function createWrapper({ propsData } = {}) {
  const node = document.createElement('div');
  document.body.appendChild(node);
  return shallowMount(AppBar, {
    store,
    propsData,
    attachTo: node,
  });
}

describe('app bar component', () => {
  describe('smoke test', () => {
    it('should render', () => {
      const wrapper = createWrapper({ loading: false });
      expect(wrapper.find({ name: 'AppBar' }).element).toBeVisible();
    });
  });
});
