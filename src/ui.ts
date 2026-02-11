// UI — minimal DOM wiring for game picker, variant select, status text

import { GAMES, getModel } from './models'

export type UIElements = {
  gameSelect: HTMLSelectElement
  variantSelect: HTMLSelectElement
  startBtn: HTMLButtonElement
  canvas: HTMLCanvasElement
  statusEl: HTMLElement
  fpsEl: HTMLElement
  keysHintEl: HTMLElement
}

export function getUI(): UIElements {
  return {
    gameSelect: document.getElementById('game-select') as HTMLSelectElement,
    variantSelect: document.getElementById('variant-select') as HTMLSelectElement,
    startBtn: document.getElementById('start-btn') as HTMLButtonElement,
    canvas: document.getElementById('game-canvas') as HTMLCanvasElement,
    statusEl: document.getElementById('status') as HTMLElement,
    fpsEl: document.getElementById('fps') as HTMLElement,
    keysHintEl: document.getElementById('keys-hint') as HTMLElement,
  }
}

export function populateGameSelect(ui: UIElements) {
  const { gameSelect, variantSelect } = ui

  for (const game of GAMES) {
    const opt = document.createElement('option')
    opt.value = game.id
    opt.textContent = game.name
    gameSelect.appendChild(opt)
  }

  function updateVariants() {
    const gameId = gameSelect.value
    const game = GAMES.find(g => g.id === gameId)!
    variantSelect.innerHTML = ''
    for (const v of game.variants) {
      const opt = document.createElement('option')
      opt.value = v.modelId
      opt.textContent = v.label
      variantSelect.appendChild(opt)
    }
    variantSelect.value = game.defaultVariant
    updateKeysHint(ui, game.defaultVariant)
  }

  variantSelect.addEventListener('change', () => {
    updateKeysHint(ui, variantSelect.value)
  })

  gameSelect.addEventListener('change', updateVariants)
  updateVariants()
}

function updateKeysHint(ui: UIElements, modelId: string) {
  const config = getModel(modelId)
  const keys = Object.keys(config.keymap).filter(k => k.length === 1 || k.startsWith('Arrow'))
  const unique = [...new Set(keys.map(k => {
    if (k === ' ') return 'Space'
    if (k.startsWith('Arrow')) return k.replace('Arrow', '')
    return k.toUpperCase()
  }))]
  ui.keysHintEl.textContent = `Controls: ${unique.join('  ')}`
}

export function getSelectedModelId(ui: UIElements): string {
  return ui.variantSelect.value
}

export function setStatus(ui: UIElements, msg: string) {
  ui.statusEl.textContent = msg
}
