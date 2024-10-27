package mat

import (
	"errors"
	"math"
	"math/rand/v2"
	"reflect"
	"sync"
)

type Mat struct {
	shape []int
	data  []float64
}

var ErrMatShapeMismatch = errors.New("matrix shape mismatch")

func GetSizeOfShape(shape []int) int {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return size
}

func flatten(input interface{}) []float64 {
	flatData := []float64{}

	switch reflect.TypeOf(input).Kind() {
	case reflect.Slice:
		v := reflect.ValueOf(input)
		for i := 0; i < v.Len(); i++ {
			flatData = append(flatData, flatten(v.Index(i).Interface())...)
		}
	case reflect.Float64:
		flatData = append(flatData, input.(float64))
	}

	return flatData
}

func calculateShape(input interface{}) []int {
	shape := []int{}
	val := reflect.ValueOf(input)

	for val.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		val = val.Index(0)
	}

	return shape
}
func MultiplyHigherDimMatricesUpdated(matA, matB *Mat) (*Mat, error) {
	if len(matA.shape) < 2 || len(matB.shape) < 2 || matA.shape[len(matA.shape)-1] != matB.shape[len(matB.shape)-2] {
		return &Mat{}, ErrMatShapeMismatch
	}

	outerDimsA := make([]int, len(matA.shape)-2)
	copy(outerDimsA, matA.shape[:len(matA.shape)-2])
	resultShape := append(outerDimsA, matA.shape[len(matA.shape)-2], matB.shape[len(matB.shape)-1])
	resultData := make([]float64, GetSizeOfShape(resultShape))
	outerSize := GetSizeOfShape(outerDimsA)

	innerDimA := matA.shape[len(matA.shape)-2]
	innerDimB := matB.shape[len(matB.shape)-1]
	sharedDim := matA.shape[len(matA.shape)-1]
	wg := sync.WaitGroup{}

	for idx := 0; idx < outerSize; idx++ {
		offsetA := idx * innerDimA * sharedDim
		offsetB := idx * sharedDim * innerDimB
		offsetResult := idx * innerDimA * innerDimB

		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < innerDimA; i++ {
				for j := 0; j < innerDimB; j++ {
					sum := 0.0
					for k := 0; k < sharedDim; k++ {
						indexA := offsetA + i*sharedDim + k
						indexB := offsetB + k*innerDimB + j
						sum += matA.data[indexA] * matB.data[indexB]
					}
					resultData[offsetResult+i*innerDimB+j] = sum
				}
			}
		}()
	}

	wg.Wait()
	return &Mat{shape: resultShape, data: resultData}, nil
}
func NewMat(input interface{}) *Mat {
	shape := calculateShape(input)
	data := flatten(input)
	return &Mat{shape: shape, data: data}
}
func NewMatFromData(shape []int, data []float64) *Mat {
	return &Mat{shape, data}
}
func Random(shape []int) *Mat {
	size := GetSizeOfShape(shape)
	data := make([]float64, size)

	for i := 0; i < size; i++ {
		data[i] = rand.Float64()
	}

	return &Mat{shape, data}
}
func MatZeros(shape []int) *Mat {
	data := make([]float64, GetSizeOfShape(shape))
	return &Mat{shape, data}
}
func (m *Mat) SameShape(n *Mat) bool {
	if len(m.shape) != len(n.shape) {
		return false
	}

	for i := range m.shape {
		if m.shape[i] != n.shape[i] {
			return false
		}
	}
	return true
}
func (m *Mat) Shape() []int {
	return m.shape
}
func (m *Mat) Size() int {
	return GetSizeOfShape(m.shape)
}
func (m *Mat) Raw() []float64 {
	return m.data
}
func (m *Mat) loc(idx []int) int {
	ret := 0
	for i := 0; i < len(idx); i++ {
		ret = ret*m.shape[i] + idx[i]
	}
	return ret
}
func (m *Mat) At(idx ...int) float64 {
	return m.data[m.loc(idx)]
}
func (m *Mat) Set(val float64, idx ...int) {
	m.data[m.loc(idx)] = val
}
func (m *Mat) Add(n *Mat) *Mat {
	if !m.SameShape(n) {
		if _, err := canBroadcast(m.shape, n.shape); err != nil {
			panic(ErrMatShapeMismatch)
		} else {
			var err error
			m, n, err = doBroadcast2(m, n)
			if err != nil {
				panic(ErrMatShapeMismatch)
			}
		}
	}

	ret := MatZeros(m.shape)
	for i := 0; i < len(m.data); i++ {
		ret.data[i] = m.data[i] + n.data[i]
	}

	return ret
}
func (m *Mat) Sub(n *Mat) *Mat {
	return m.Add(n.ScalarMul(-1))
}
func (m *Mat) Batch(idx int) *Mat {
	if len(m.shape) == 1 {
		return &Mat{[]int{1}, []float64{m.data[idx]}}
	}

	shape := m.shape[1:]
	start := idx * GetSizeOfShape(shape)
	end := (idx + 1) * GetSizeOfShape(shape)

	return &Mat{shape, m.data[start:end]}
}
func (m *Mat) SetBatch(n *Mat, idx int) {
	shape := m.shape[1:]

	temp := &Mat{shape, nil}
	if !temp.SameShape(n) {
		panic(ErrMatShapeMismatch)
	}
	copy(m.data[idx*GetSizeOfShape(shape):(idx+1)*GetSizeOfShape(shape)], n.data)
}
func (a *Mat) Mul(b *Mat) *Mat {
	for len(a.shape) != len(b.shape) {
		if len(a.shape) < len(b.shape) {
			a = a.Wrap()
		} else {
			b = b.Wrap()
		}
	}

	if len(a.shape) == 1 {
		panic(ErrMatShapeMismatch)
	}

	for i := 0; i <= len(a.shape)-3; i++ {
		if a.shape[i] != b.shape[i] {
			panic(ErrMatShapeMismatch)
		}
	}

	n := a.shape[len(a.shape)-1]
	p := b.shape[len(b.shape)-2]
	if n != p {
		println(n, p)
		panic(ErrMatShapeMismatch)
	}

	if len(a.shape) == 2 {
		m, n, p := a.shape[0], a.shape[1], b.shape[1]
		res := make([]float64, m*p)
		for i := 0; i < m; i++ {
			for j := 0; j < p; j++ {
				sum := 0.0
				for k := 0; k < n; k++ {
					sum += a.At(i, k) * b.At(k, j)
				}
				res[i*p+j] = sum
			}
		}

		return &Mat{[]int{m, p}, res}
	}

	res, err := MultiplyHigherDimMatricesUpdated(a, b)
	if err != nil {
		panic(err)
	}

	return res
}
func (m *Mat) Print() {
	printMat(m, 0)
}
func printIndent(indent int) {
	for i := 0; i < indent; i++ {
		print("  ")
	}
}
func printMat(m *Mat, indent int) {
	if len(m.shape) == 1 {
		printIndent(indent)
		print("[")
		for j := 0; j < m.shape[0]; j++ {
			print(m.At(j), " ")
		}
		println("]")
		return
	}
	if len(m.shape) == 2 {
		printIndent(indent)
		println("[")
		for i := 0; i < m.shape[0]; i++ {
			printIndent(indent + 1)
			print("[")
			for j := 0; j < m.shape[1]; j++ {
				print(m.At(i, j), " ")
			}
			println("]")
		}
		printIndent(indent)
		println("]")
		return
	}
	printIndent(indent)
	println("[")
	for i := 0; i < m.shape[0]; i++ {
		printMat(m.Batch(i), indent+1)
	}
	printIndent(indent)
	println("]")
}
func (m *Mat) Transpose(axes []int) *Mat {
	if len(axes) != len(m.shape) {
		panic(ErrMatShapeMismatch)
	}

	newShape := make([]int, len(axes))
	for i, axis := range axes {
		newShape[i] = m.shape[axis]
	}

	newData := make([]float64, len(m.data))
	transposeRecursive(m, newData, []int{}, []int{}, axes)

	return &Mat{newShape, newData}
}
func (m *Mat) T() *Mat {
	if len(m.shape) != 2 {
		panic(ErrMatShapeMismatch)
	}

	return m.Transpose([]int{1, 0})
}
func (m *Mat) Reshape(newShape ...int) *Mat {
	oldSize := GetSizeOfShape(m.shape)
	X := 1
	negIdx := -1

	for i, dim := range newShape {
		if dim == -1 {
			if negIdx != -1 {
				panic(ErrMatShapeMismatch)
			}
			negIdx = i
			break
		} else {
			X *= dim
		}
	}
	if negIdx != -1 {
		if oldSize%X != 0 {
			panic(ErrMatShapeMismatch)
		}

		newShape = newShape[:negIdx+1]
		newShape[negIdx] = oldSize / X
	} else if X != oldSize {
		panic(ErrMatShapeMismatch)
	}
	return &Mat{shape: newShape, data: m.data}
}
func transposeRecursive(m *Mat, newData []float64, oldIdx, newIdx, axes []int) {
	if len(oldIdx) == len(m.shape) {
		oldLoc := m.loc(oldIdx)
		newLoc := m.loc(newIdx)
		newData[newLoc] = m.data[oldLoc]
		return
	}

	dim := len(oldIdx)
	for i := 0; i < m.shape[dim]; i++ {
		transposeRecursive(m, newData, append(oldIdx, i), append(newIdx, i), axes)
	}
}
func (m *Mat) Flatten() *Mat {
	res := m.Reshape(-1)
	return res
}
func (m *Mat) ScalarAdd(n float64) *Mat {
	data := make([]float64, len(m.data))

	for i := range m.data {
		data[i] = m.data[i] + n
	}

	return &Mat{m.shape, data}
}
func (m *Mat) ScalarSub(n float64) *Mat {
	return m.ScalarAdd(-n)
}
func (m *Mat) ScalarMul(n float64) *Mat {
	data := make([]float64, len(m.data))

	for i := range m.data {
		data[i] = m.data[i] * n
	}

	return &Mat{m.shape, data}
}
func (m *Mat) ScalarDiv(n float64) *Mat {
	return m.ScalarMul(1 / n)
}
func (m *Mat) AddElem(n *Mat) *Mat {
	if !m.SameShape(n) {
		if _, err := canBroadcast(m.shape, n.shape); err != nil {
			panic(ErrMatShapeMismatch)
		} else {
			var err error
			m, n, err = doBroadcast2(m, n)
			if err != nil {
				panic(ErrMatShapeMismatch)
			}
		}
	}

	data := make([]float64, len(m.data))
	for i := range m.data {
		data[i] = m.data[i] + n.data[i]
	}

	return &Mat{m.shape, data}
}
func (m *Mat) SubElem(n *Mat) *Mat {
	return m.AddElem(n.ScalarMul(-1))
}
func (m *Mat) MulElem(n *Mat) *Mat {
	if !m.SameShape(n) {
		if _, err := canBroadcast(m.shape, n.shape); err != nil {
			panic(ErrMatShapeMismatch)
		} else {
			var err error
			m, n, err = doBroadcast2(m, n)
			if err != nil {
				panic(ErrMatShapeMismatch)
			}
		}
	}

	data := make([]float64, len(m.data))
	for i := range m.data {
		data[i] = m.data[i] * n.data[i]
	}

	return &Mat{m.shape, data}
}
func (m *Mat) DivElem(n *Mat) *Mat {
	if !m.SameShape(n) {
		if _, err := canBroadcast(m.shape, n.shape); err != nil {
			panic(ErrMatShapeMismatch)
		} else {
			var err error
			m, n, err = doBroadcast2(m, n)
			if err != nil {
				panic(ErrMatShapeMismatch)
			}
		}
	}

	data := make([]float64, len(m.data))
	for i := range m.data {
		data[i] = m.data[i] / n.data[i]
	}

	return &Mat{m.shape, data}
}
func (m *Mat) Pow(n float64) *Mat {
	data := make([]float64, len(m.data))

	for i := range m.data {
		data[i] = math.Pow(m.data[i], n)
	}

	return &Mat{m.shape, data}
}
func (m *Mat) Map(fn func(float64) float64) *Mat {
	data := make([]float64, len(m.data))
	for i := range m.data {
		data[i] = fn(m.data[i])
	}
	return &Mat{m.shape, data}
}
func (m *Mat) Squeeze() *Mat {
	newShape := []int{}

	for _, dim := range m.shape {
		if dim != 1 {
			newShape = append(newShape, dim)
		}
	}

	return &Mat{shape: newShape, data: m.data}
}
func (m *Mat) ArgMax(axis int) []int {
	if axis >= len(m.shape) {
		panic("Axis is out of bounds for array shape")
	}

	outShape := make([]int, len(m.shape)-1)
	outIdx := 0
	for i := 0; i < len(m.shape); i++ {
		if i != axis {
			outShape[outIdx] = m.shape[i]
			outIdx++
		}
	}

	resultSize := GetSizeOfShape(outShape)
	result := make([]int, resultSize)
	for i := 0; i < resultSize; i++ {
		maxVal := -1.0
		maxIdx := 0
		for j := 0; j < m.shape[axis]; j++ {
			idx := m.locOf(i, j, axis)
			if m.data[idx] > maxVal {
				maxVal = m.data[idx]
				maxIdx = j
			}
		}
		result[i] = maxIdx
	}

	return result
}
func (m *Mat) locOf(flatIdx, subIdx, axis int) int {
	stride := 1
	for i := axis + 1; i < len(m.shape); i++ {
		stride *= m.shape[i]
	}
	return flatIdx*stride + subIdx*stride
}
func (m *Mat) Wrap() *Mat {
	shape := []int{1}
	shape = append(shape, m.shape...)
	return &Mat{shape, m.data}
}
func (m *Mat) Unwrap() *Mat {
	shape := m.shape[1:]
	return &Mat{shape, m.data}
}
func canBroadcast(shapeA, shapeB []int) ([]int, error) {
	lenA, lenB := len(shapeA), len(shapeB)
	maxLen := lenA
	if lenB > maxLen {
		maxLen = lenB
	}

	resultShape := make([]int, maxLen)
	for i := 0; i < maxLen; i++ {
		dimA, dimB := 1, 1
		if i < lenA {
			dimA = shapeA[lenA-1-i]
		}
		if i < lenB {
			dimB = shapeB[lenB-1-i]
		}
		if dimA != dimB && dimA != 1 && dimB != 1 {
			return nil, ErrMatShapeMismatch
		}
		resultShape[maxLen-1-i] = max(dimA, dimB)
	}
	return resultShape, nil
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func doBroadcastTo(input *Mat, targetShape []int) (*Mat, error) {
	inputShape := input.Shape()

	if len(inputShape) > len(targetShape) {
		return nil, errors.New("input array has more dimensions than target shape")
	}

	for len(inputShape) < len(targetShape) {
		inputShape = append([]int{1}, inputShape...)
	}
	for i := 0; i < len(targetShape); i++ {
		if inputShape[i] != targetShape[i] && inputShape[i] != 1 {
			return nil, errors.New("shapes are not compatible for broadcasting")
		}
	}

	outputData := make([]float64, GetSizeOfShape(targetShape))
	outputMat := &Mat{shape: targetShape, data: outputData}
	fillBroadcastData(input, outputMat)

	return outputMat, nil
}

func GetIndicesFromFlatIndex(flatIndex int, shape []int) []int {
	indices := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		indices[i] = flatIndex % shape[i]
		flatIndex /= shape[i]
	}
	return indices
}

func fillBroadcastData(input, output *Mat) {
	inputShape := input.Shape()

	for i := 0; i < len(output.data); i++ {
		inputIdx := make([]int, len(inputShape))
		outputIdx := GetIndicesFromFlatIndex(i, output.Shape())

		// 맞는 차원 인덱스로 매핑
		for j := range inputShape {
			if inputShape[j] == 1 {
				inputIdx[j] = 0
			} else {
				inputIdx[j] = outputIdx[j]
			}
		}

		// 값 설정
		output.Set(input.At(inputIdx...), outputIdx...)
	}
}
func doBroadcast(inputs ...*Mat) ([]*Mat, error) {
	if len(inputs) == 0 {
		return nil, errors.New("no input arrays provided")
	}

	targetShape := inputs[0].Shape()
	for _, input := range inputs[1:] {
		targetShape, _ = findBroadcastShape(targetShape, input.Shape())
	}

	broadcastedArrays := make([]*Mat, len(inputs))
	for i, input := range inputs {
		broadcasted, err := doBroadcastTo(input, targetShape)
		if err != nil {
			return nil, err
		}
		broadcastedArrays[i] = broadcasted
	}

	return broadcastedArrays, nil
}
func doBroadcast2(a, b *Mat) (*Mat, *Mat, error) {
	res, err := doBroadcast(a, b)
	if err != nil {
		return nil, nil, err
	}
	return res[0], res[1], nil
}
func findBroadcastShape(shapeA, shapeB []int) ([]int, error) {
	if len(shapeA) < len(shapeB) {
		shapeA, shapeB = shapeB, shapeA
	}

	resultShape := make([]int, len(shapeA))
	copy(resultShape, shapeA)
	for i := 1; i <= len(shapeB); i++ {
		if shapeA[len(shapeA)-i] == 1 {
			resultShape[len(resultShape)-i] = shapeB[len(shapeB)-i]
		} else if shapeB[len(shapeB)-i] != 1 && shapeA[len(shapeA)-i] != shapeB[len(shapeB)-i] {
			return nil, errors.New("shapes are not compatible for broadcasting")
		}
	}

	return resultShape, nil
}
func (m *Mat) Sum() float64 {
	total := 0.0

	for _, val := range m.data {
		total += val
	}

	return total
}
func (m *Mat) SumWithAxis(axis int, keepDim bool) *Mat {
	shape := m.Shape()
	if axis >= len(shape) || axis < 0 {
		panic("Invalid axis provided")
	}

	if axis < 0 {
		axis += len(shape)
	}

	newShape := append([]int{}, shape...)
	if keepDim {
		newShape[axis] = 1 // Keep the reduced dimension with size 1
	} else {
		newShape = append(newShape[:axis], newShape[axis+1:]...) // Remove the axis
	}

	result := make([]float64, GetSizeOfShape(newShape))
	index := make([]int, len(shape))
	recursiveSumHelper(m.data, result, index, shape, newShape, axis)

	return NewMatFromData(newShape, result) // Return the new matrix with the reduced shape
}
func recursiveSumHelper(input, output []float64, index, shape, newShape []int, axis int) {
	strides := calculateStrides(shape)
	newStrides := calculateStrides(newShape)
	var getInputOffset = func(baseIndex []int) int {
		offset := 0
		for i := 0; i < len(baseIndex); i++ {
			offset += baseIndex[i] * strides[i]
		}
		return offset
	}
	var getOutputOffset = func(baseIndex []int) int {
		offset := 0
		for i := 0; i < len(baseIndex); i++ {
			offset += baseIndex[i] * newStrides[i]
		}
		return offset
	}

	for i := 0; i < GetSizeOfShape(shape[:axis]); i++ {
		for j := 0; j < GetSizeOfShape(shape[axis+1:]); j++ {
			sum := 0.0
			outputIndex := getOutputOffset(index)

			for k := 0; k < shape[axis]; k++ {
				index[axis] = k
				inputIndex := getInputOffset(index)
				sum += input[inputIndex]
			}

			output[outputIndex] = sum
			index[axis] = 0
		}
	}
}
func calculateStrides(shape []int) []int {
	strides := make([]int, len(shape))
	strides[len(shape)-1] = 1

	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}

	return strides
}
func (m *Mat) Copy() *Mat {
	data := make([]float64, len(m.data))
	copy(data, m.data)

	return &Mat{m.shape, data}
}
