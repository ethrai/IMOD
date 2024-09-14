package perceptron

import (
	"fmt"
	"math/rand/v2"
	"reflect"
)

// Perceptron is a simple one-layer perceptron with binary step activation
// function. It takes two inputs and outputs a single value.
type Perceptron struct {
	weights   []float64
	rate      float64
	threshold float64
}

func New(rate float64) Perceptron {
	weights := []float64{rand.Float64(), rand.Float64()}
	threshold := rand.Float64()
	return Perceptron{weights, rate, threshold}
}

// Actiavte takes a set of inputs and returns the output of the perceptron.
func (p *Perceptron) Activate(input []float64) float64 {
	sum := p.dotProduct(input)
	return calcOut(sum)
}

// Learn takes a set of inputs and expected outputs and updates the weights.
func (p *Perceptron) Learn(inputs [][]float64, expected []float64) {
	outputs := make([]float64, len(inputs))
	var c int
	for !reflect.DeepEqual(outputs, expected) {
		c++
		fmt.Printf("iteration %d\n", c)
		for i := range inputs { // Calculate output (y) for each input
			sum := p.dotProduct(inputs[i])
			outputs[i] = calcOut(sum)
			for j := range p.weights {
				p.weights[j] -= p.rate * inputs[i][j] * (outputs[i] - expected[i])
			}
			p.threshold += p.rate * (outputs[i] - expected[i])
			fmt.Printf("weights: %v\n", p.weights)
			fmt.Printf("threshold: %v\n", p.threshold)
		}
	}
}

// dotProduct calculates the dot product of the input and the weights.
func (p Perceptron) dotProduct(input []float64) float64 {
	var sum float64
	for i, w := range p.weights {
		sum += w * input[i]
	}
	return sum - p.threshold
}

func calcOut(sum float64) float64 {
	if sum >= 0 {
		return 1
	}
	return -1
}
