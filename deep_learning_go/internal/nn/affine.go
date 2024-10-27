package nn

import (
	"deep_learning_go/internal/mat/v2"
	"math"
)

type Affine struct {
	w  *mat.Mat
	b  *mat.Mat
	dW *mat.Mat
	dB *mat.Mat
	x  *mat.Mat
}

func XavierInit(shape ...int) *mat.Mat {
	scale := math.Sqrt(2.0 / float64(shape[0]+shape[1]))
	return mat.Random(shape).ScalarMul(scale)
}

func NewAffine(inputSize, outputSize int) *Affine {
	W := XavierInit(inputSize, outputSize)
	b := mat.MatZeros([]int{1, outputSize})

	return &Affine{
		w: W,
		b: b,
	}
}

func (a *Affine) Forward(x *mat.Mat) *mat.Mat {
	a.x = x

	return a.x.Mul(a.w).Add(a.b)
}

func (a *Affine) Backward(dout *mat.Mat) *mat.Mat {
	a.dW = a.x.T().Mul(dout)
	a.dB = dout.SumWithAxis(0, true)

	dx := dout.Mul(a.w.T())

	return dx
}

func (a *Affine) W() *mat.Mat {
	return a.w
}
func (a *Affine) B() *mat.Mat {
	return a.b
}
func (a *Affine) DW() *mat.Mat {
	return a.dW
}
func (a *Affine) DB() *mat.Mat {
	return a.dB
}

func (a *Affine) SetW(w *mat.Mat) {
	a.w = w
}
func (a *Affine) SetB(b *mat.Mat) {
	a.b = b
}
func (a *Affine) SetDW(dw *mat.Mat) {
	a.dW = dw
}
func (a *Affine) SetDB(db *mat.Mat) {
	a.dB = db
}
