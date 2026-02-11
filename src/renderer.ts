// Renderer — quantize float32 to ImageData + draw to canvas
// OffscreenCanvas + ImageData created once, reused forever

export class Renderer {
  private ctx: CanvasRenderingContext2D
  private canvasW: number
  private canvasH: number
  private offscreen: OffscreenCanvas | null = null
  private offCtx: OffscreenCanvasRenderingContext2D | null = null
  private imageData: ImageData | null = null
  private lastW = 0
  private lastH = 0

  constructor(canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!
    this.canvasW = canvas.width
    this.canvasH = canvas.height
  }

  draw(buf: Float32Array, c: number, h: number, w: number) {
    if (!this.offscreen || this.lastW !== w || this.lastH !== h) {
      this.offscreen = new OffscreenCanvas(w, h)
      this.offCtx = this.offscreen.getContext('2d')!
      this.imageData = new ImageData(w, h)
      this.lastW = w
      this.lastH = h
    }

    quantizeToImageData(buf, c, h, w, this.imageData!)
    this.offCtx!.putImageData(this.imageData!, 0, 0)
    this.ctx.drawImage(this.offscreen!, 0, 0, this.canvasW, this.canvasH)
  }
}

function quantizeToImageData(src: Float32Array, _c: number, h: number, w: number, dst: ImageData) {
  const out = dst.data
  const planeSize = h * w

  if (dst.width === w && dst.height === h) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x
        const di = idx * 4
        out[di]     = Math.max(0, Math.min(255, ((src[idx] + 1) * 127.5) | 0))
        out[di + 1] = Math.max(0, Math.min(255, ((src[planeSize + idx] + 1) * 127.5) | 0))
        out[di + 2] = Math.max(0, Math.min(255, ((src[2 * planeSize + idx] + 1) * 127.5) | 0))
        out[di + 3] = 255
      }
    }
  } else {
    const scaleY = h / dst.height
    const scaleX = w / dst.width
    for (let y = 0; y < dst.height; y++) {
      const sy = (y * scaleY) | 0
      for (let x = 0; x < dst.width; x++) {
        const sx = (x * scaleX) | 0
        const idx = sy * w + sx
        const di = (y * dst.width + x) * 4
        out[di]     = Math.max(0, Math.min(255, ((src[idx] + 1) * 127.5) | 0))
        out[di + 1] = Math.max(0, Math.min(255, ((src[planeSize + idx] + 1) * 127.5) | 0))
        out[di + 2] = Math.max(0, Math.min(255, ((src[2 * planeSize + idx] + 1) * 127.5) | 0))
        out[di + 3] = 255
      }
    }
  }
}
