package predictive

import (
	"reflect"
	"testing"
)

func fn(n float64) float64 {
	if n <= 1 {
		return n
	}
	return fn(n-1) + fn(n-2)
}

func Test_PredictiveModel(t *testing.T) {
	model := New(0.5, 3)

	trainingSize := 15
	predictionSize := 5

	const (
		a = 2
		b = 9
		d = 0.4
	)

	tabRes := make([]float64, trainingSize)
	for i := 0; i < trainingSize; i++ {
		tabRes[i] = fn(float64(i))
	}

	model.Learn(tabRes)

	t.Run("predicting data", func(t *testing.T) {
		want := []float64{13, 21, 34, 55, 89}

		in := []float64{0, 1, 1, 2, 3, 5, 8}
		actual := model.Predict(in, predictionSize)

		if !reflect.DeepEqual(actual, want) {
			t.Errorf("actual: %v, want: %v", actual, want)
		}
	})
}

func Test_getInputRef(t *testing.T) {
	m := New(0, 3)
	inputs := []float64{1, 2, 3, 5, 8, 13}
	matrix, refs := m.getInputRef(inputs)

	expMatrix := [][]float64{
		{1, 2, 3},
		{2, 3, 5},
		{3, 5, 8},
	}

	expRefs := []float64{5, 8, 13}

	if !reflect.DeepEqual(matrix, expMatrix) {
		t.Errorf("actual matrix: %v, expected: %v", matrix, expMatrix)
	}

	if !reflect.DeepEqual(refs, expRefs) {
		t.Errorf("actual refs: %v, expected refs: %v", refs, expRefs)
	}
}
