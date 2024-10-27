package nn

import "deep_learning_go/internal/mat/v2"

type Param interface {
	W() *mat.Mat
	B() *mat.Mat
	DW() *mat.Mat
	DB() *mat.Mat

	SetW(w *mat.Mat)
	SetB(b *mat.Mat)
	SetDW(dw *mat.Mat)
	SetDB(db *mat.Mat)
}
type Layer interface {
	Forward(x *mat.Mat) *mat.Mat
	Backward(dout *mat.Mat) *mat.Mat
}
