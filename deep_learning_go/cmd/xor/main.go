package main

import (
	"deep_learning_go/internal/mat/v2"
	"deep_learning_go/internal/nn"
	"fmt"
)

func main() {
	xorInputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	xorOutputs := mat.NewMat([]float64{0, 1, 1, 0}).Wrap().T()

	input := mat.NewMat(xorInputs)

	affine1 := nn.NewAffine(2, 4)
	relu := nn.NewRelu()
	affine2 := nn.NewAffine(4, 1)
	sigmoid := nn.NewSigmoid()

	loss_layer := nn.NewBinaryCrossEntropy()

	adam := nn.NewAdam(0.0001, 0.9, 0.999, 1e-7)

	N := 10
	// 학습 과정
	for epoch := 0; epoch < N; epoch++ {
		// 순전파
		out1 := affine1.Forward(input)
		reluOut := relu.Forward(out1)
		out2 := affine2.Forward(reluOut)
		pred := sigmoid.Forward(out2)

		// 손실 계산 (Binary Cross-Entropy)
		loss := loss_layer.Forward(pred, xorOutputs)
		fmt.Printf("Epoch %d, Loss: %f\n", epoch, loss)

		// 역전파
		dpred := loss_layer.Backward(pred, xorOutputs)
		dout2 := sigmoid.Backward(dpred)
		dout1 := affine2.Backward(dout2)
		drelu := relu.Backward(dout1)
		_ = affine1.Backward(drelu)

		// 가중치 업데이트
		adam.Update(affine2)
		adam.Update(affine1)

		// 각 단계의 기울기 확인
		fmt.Println("dpred:")
		dpred.Print()
		fmt.Println("dout2:")
		dout2.Print()
		fmt.Println("Affine2 dW:")
		affine2.DW().Print()
		affine2.DB().Print()
		fmt.Println("drelu:")
		drelu.Print()
		fmt.Println("Affine1 dW:")
		affine1.DW().Print()
		fmt.Println("Affine1 dB:")
		affine1.DB().Print()

		fmt.Printf("Epoch %d / %d (%.2f%%), Loss: %f\n", epoch+1, N, float64(epoch)/float64(N)*100, loss)
	}

	println("TEST")

	// 테스트
	for i, input := range xorInputs {
		m_input := mat.NewMat(input)
		out1 := affine1.Forward(m_input)
		reluOut := sigmoid.Forward(out1)
		pred := affine2.Forward(reluOut)

		fmt.Printf("Input: %v, Answer: %f, Output: %f\n", input, xorOutputs.Raw()[i], pred.At(0))
	}
}
