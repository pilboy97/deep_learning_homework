package nn

import "deep_learning_go/internal/mat/v2"

type ReLU struct {
	x *mat.Mat
}

func NewRelu() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x *mat.Mat) *mat.Mat {
	r.x = x
	return x.Map(func(v float64) float64 {
		if v <= 0 {
			return 0
		}
		return v
	})
}

func (r *ReLU) Backward(dout *mat.Mat) *mat.Mat {
	res := dout.Copy()
	for i := 0; i < r.x.Shape()[0]; i++ {
		for j := 0; j < r.x.Shape()[1]; j++ {
			if r.x.At(i, j) <= 0 {
				res.Set(0, i, j)
			}
		}
	}
	return res
}
