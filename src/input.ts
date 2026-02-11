// Input handling — keyboard + touch zones

import { ModelConfig } from './models'

export type InputHandler = {
  getAction: () => number
  destroy: () => void
}

export function createInput(config: ModelConfig, canvas: HTMLCanvasElement): InputHandler {
  let currentAction = config.defaultAction
  const keymap = config.keymap

  // Keyboard
  function onKeyDown(e: KeyboardEvent) {
    const action = keymap[e.key]
    if (action !== undefined) {
      e.preventDefault()
      currentAction = action
    }
  }
  function onKeyUp(e: KeyboardEvent) {
    const action = keymap[e.key]
    if (action !== undefined && currentAction === action) {
      currentAction = config.defaultAction
    }
  }
  window.addEventListener('keydown', onKeyDown)
  window.addEventListener('keyup', onKeyUp)

  // Touch: divide canvas into zones
  // For 2-action games (hold): whole canvas = action 1
  // For multi-action: left third = action mapping varies by game
  const wrap = canvas.parentElement!

  function onTouchStart(e: TouchEvent) {
    e.preventDefault()
    const rect = wrap.getBoundingClientRect()
    const touch = e.touches[0]
    const relX = (touch.clientX - rect.left) / rect.width

    if (config.numActions === 2) {
      // Simple hold game
      currentAction = 1
    } else if (config.numActions <= 4) {
      // Knightfall-style: left=attack, center=block, right=magic
      if (relX < 0.33) currentAction = 1
      else if (relX < 0.66) currentAction = 2
      else currentAction = 3
    } else {
      // Anamnesis/Doom-style: left=turn_left, center=forward, right=turn_right
      if (relX < 0.33) {
        // Find a "left" action: turn_left or left_rotation
        currentAction = keymap['ArrowLeft'] ?? keymap['a'] ?? 2
      } else if (relX < 0.66) {
        currentAction = keymap['ArrowUp'] ?? keymap['w'] ?? 1
      } else {
        currentAction = keymap['ArrowRight'] ?? keymap['d'] ?? 3
      }
    }
  }

  function onTouchEnd(e: TouchEvent) {
    e.preventDefault()
    if (e.touches.length === 0) {
      currentAction = config.defaultAction
    }
  }

  wrap.addEventListener('touchstart', onTouchStart, { passive: false })
  wrap.addEventListener('touchend', onTouchEnd, { passive: false })
  wrap.addEventListener('touchcancel', onTouchEnd, { passive: false })

  return {
    getAction: () => currentAction,
    destroy() {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      wrap.removeEventListener('touchstart', onTouchStart)
      wrap.removeEventListener('touchend', onTouchEnd)
      wrap.removeEventListener('touchcancel', onTouchEnd)
    }
  }
}
