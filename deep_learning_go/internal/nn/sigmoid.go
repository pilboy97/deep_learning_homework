package nn

import (
	"deep_learning_go/internal/mat/v2"
	"math"
)

type Sigmoid struct {
	out *mat.Mat
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(x *mat.Mat) *mat.Mat {
	s.out = x.Map(func(v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}).Map(func(p float64) float64 {
		epsilon := 1e-7
		if p < epsilon {
			return epsilon
		} else if p > 1-epsilon {
			return 1 - epsilon
		}
		return p
	})

	return s.out
}

func (s *Sigmoid) Backward(dout *mat.Mat) *mat.Mat {
	return s.out.MulElem(s.out.ScalarSub(1)).MulElem(dout)
}
