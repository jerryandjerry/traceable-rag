import { proxy } from 'valtio'

const state = proxy({
  list: [] as API.Session[],
  // Off delegates the search decision to intent detection; on requires web
  // search for that message.
  useWeb: false,
  useDeep: false,
})

const actions = {
  setList(list: API.Session[]) {
    state.list = list
  },
  add(item: API.Session) {
    state.list.push(item)
  },
  setUseWeb(useWeb: boolean) {
    state.useWeb = useWeb
  },

  setUseDeep(useDeep: boolean) {
    state.useDeep = useDeep
  },
}

export const sessionState = state
export const sessionActions = actions
