package mat

import (
	"reflect"
	"testing"
)

func TestMultiplyHigherDimMatrices(t *testing.T) {
	matA := &Mat{
		shape: []int{2, 3},
		data:  []float64{1, 2, 3, 4, 5, 6},
	}
	matB := &Mat{
		shape: []int{3, 2},
		data:  []float64{7, 8, 9, 10, 11, 12},
	}
	expected2D := &Mat{
		shape: []int{2, 2},
		data:  []float64{58, 64, 139, 154},
	}

	result2D := matA.Mul(matB)
	if !reflect.DeepEqual(result2D, expected2D) {
		t.Errorf("2D multiplication result mismatch, got: %v, want: %v", result2D, expected2D)
	}

	matA3D := &Mat{
		shape: []int{1, 2, 3},
		data:  []float64{1, 2, 3, 4, 5, 6},
	}
	matB3D := &Mat{
		shape: []int{1, 3, 2},
		data:  []float64{7, 8, 9, 10, 11, 12},
	}
	expected3D := &Mat{
		shape: []int{1, 2, 2},
		data:  []float64{58, 64, 139, 154},
	}

	result3D := matA3D.Mul(matB3D)
	if !reflect.DeepEqual(result3D, expected3D) {
		t.Errorf("3D multiplication result mismatch, got: %v, want: %v", result3D, expected3D)
	}
}
