package nn

import (
	"deep_learning_go/internal/mat/v2"
	"math"
)

type BinaryCrossEntropy struct {
}

func NewBinaryCrossEntropy() *BinaryCrossEntropy {
	return &BinaryCrossEntropy{}
}

func (b *BinaryCrossEntropy) Forward(pred, target *mat.Mat) float64 {
	epsilon := 1e-7 // log(0) 오류 방지를 위한 작은 값 추가

	clampedPred := pred.Map(func(p float64) float64 {
		if p < epsilon {
			return epsilon
		} else if p > 1-epsilon {
			return 1 - epsilon
		}
		return p
	})

	oneMinusClampedPred := clampedPred.ScalarSub(1).Map(func(p float64) float64 {
		if p < epsilon {
			return epsilon
		}
		return p
	})

	// Binary Cross-Entropy 계산
	loss := clampedPred.Map(math.Log).MulElem(target).ScalarMul(-1).Sub(
		target.ScalarSub(1).MulElem(oneMinusClampedPred.Map(math.Log)).ScalarMul(-1),
	).Sum()

	return loss / float64(target.Shape()[0]) // 샘플 개수로 나누어 평균 손실 계산
}

func (b *BinaryCrossEntropy) Backward(pred, target *mat.Mat) *mat.Mat {
	epsilon := 1e-7 // log(0) 오류 방지를 위한 작은 값 추가

	clampedPred := pred.Map(func(p float64) float64 {
		if p < epsilon {
			return epsilon
		} else if p > 1-epsilon {
			return 1 - epsilon
		}
		return p
	})

	grad := target.DivElem(clampedPred).ScalarMul(-1).Add(
		target.ScalarSub(1).DivElem(clampedPred.ScalarSub(1)),
	)

	return grad
}
