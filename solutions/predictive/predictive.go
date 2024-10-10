package predictive

import (
	"fmt"
	"math"
	"math/rand/v2"
)

const EMin = 0.1

// PredictiveModel is perceptron that follows Delta Rule to predict next N
// values in a timestamp.
type PredictiveModel struct {
	Weights []float64
	T       float64
	Rate    float64 // learning rate
	N       int     // input / window size
}

func New(rate float64, inputSize int) *PredictiveModel {
	pm := &PredictiveModel{
		Rate: rate,
		N:    inputSize,
	}
	pm.initWeights()
	pm.T = rand.Float64()
	return pm
}

// Learn will use delta rule to calculate weights for prediction model
func (p *PredictiveModel) Learn(inputs []float64) {
	if len(inputs) == 0 {
		panic("inputs are empty")
	}

	samples, refValues := p.getInputRef(inputs)
	E := math.MaxFloat64

	for E > EMin {
		E = 0
		for i, sample := range samples {
			// output
			y := p.dotProduct(sample) - p.T
			// Update weights
			for j := range p.Weights {
				p.Weights[j] -= p.Rate * samples[i][j] * (y - refValues[i])
			}
			// Update threshold
			p.T += p.Rate * (y - refValues[i])
			// add to mean square error
			E += 0.5 * math.Pow(y-refValues[i], 2)
		}
    fmt.Println("current error: ", E)
	}
}

// Predict calculates n next values for PredictFn.
func (p *PredictiveModel) Predict(input []float64, n int) []float64 {
	result := make([]float64, n)
	for i := 0; i < len(result); i++ {
		result[i] = p.dotProduct(input) - p.T
	}

	return result
}

func (p *PredictiveModel) dotProduct(sample []float64) float64 {
	var sum float64
	for i, w := range p.Weights {
		sum += w * sample[i]
	}
	return sum
}

// getInputRef parses inputs into linear autoregression matrix and returns this
// matrix as well as wanted outcome
func (p *PredictiveModel) getInputRef(inputs []float64) ([][]float64, []float64) {
	sampleMatrix := make([][]float64, 0, 4)
	refs := make([]float64, 0, p.N)
	for i := 0; i < len(inputs)-p.N; i++ {
		row := make([]float64, 0, p.N)

		row = append(row, inputs[i:p.N+i]...)
		refs = append(refs, inputs[i+p.N])

		sampleMatrix = append(sampleMatrix, row)
	}

	return sampleMatrix, refs
}

func (p PredictiveModel) initWeights() {
	weights := make([]float64, p.N)
	for i := 0; i < p.N; i++ {
		weights[i] = rand.Float64()
	}
}
