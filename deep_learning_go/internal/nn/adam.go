package nn

import (
	"deep_learning_go/internal/mat/v2"
	"math"
)

type Adam struct {
	lr      float64
	beta1   float64
	beta2   float64
	epsilon float64

	m map[Layer][2]*mat.Mat
	v map[Layer][2]*mat.Mat
	t int
}

func NewAdam(lr, beta1, beta2, epsilon float64) *Adam {
	m := make(map[Layer][2]*mat.Mat)
	v := make(map[Layer][2]*mat.Mat)

	return &Adam{
		lr:      lr,
		beta1:   beta1,
		beta2:   beta2,
		epsilon: epsilon,
		m:       m,
		v:       v,
		t:       0,
	}
}

func (adam *Adam) GetLR() float64 {
	return adam.lr
}
func (Adam *Adam) SetLR(lr float64) {
	Adam.lr = lr
}

func (adam *Adam) Update(layer Layer) {
	adam.t++

	var param Param
	var ok bool
	if param, ok = layer.(Param); !ok {
		return
	}

	W, B := param.W(), param.B()
	grad := [2]*mat.Mat{param.DW(), param.DB()}

	if _, ok := adam.m[layer]; !ok {
		adam.m[layer] = [2]*mat.Mat{mat.MatZeros(W.Shape()), mat.MatZeros(B.Shape())}
	}
	if _, ok := adam.v[layer]; !ok {
		adam.v[layer] = [2]*mat.Mat{mat.MatZeros(W.Shape()), mat.MatZeros(B.Shape())}
	}

	decay := 0.01
	lr_t := adam.lr / (1 + decay*float64(adam.t))
	lr_t *= math.Sqrt(1-math.Pow(adam.beta2, float64(adam.t))) / (1 - math.Pow(adam.beta1, float64(adam.t)))

	m_arr := adam.m[layer]
	v_arr := adam.v[layer]

	result := [2]*mat.Mat{W, B}
	for i := 0; i < 2; i++ {
		m_arr[i] = adam.m[layer][i].ScalarMul(adam.beta1).AddElem(grad[i].ScalarMul(1 - adam.beta1))

		v_arr[i] = adam.v[layer][i].ScalarMul(adam.beta2).AddElem(grad[i].MulElem(grad[i]).ScalarMul(1 - adam.beta2))

		m_hat := m_arr[i].ScalarMul(1 / (1 - math.Pow(adam.beta1, float64(adam.t))))
		v_hat := v_arr[i].ScalarMul(1 / (1 - math.Pow(adam.beta2, float64(adam.t))))

		result[i] = result[i].SubElem(m_hat.DivElem(v_hat.Map(math.Sqrt).ScalarAdd(adam.epsilon)).ScalarMul(lr_t))
	}

	adam.m[layer] = m_arr
	adam.v[layer] = v_arr

	param.SetW(result[0])
	param.SetB(result[1])
}
