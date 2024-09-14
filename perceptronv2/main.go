package main

import (
	"fmt"

	"perceptron/perceptron"
)

func main() {
	p := perceptron.New(1e-1)
	dataset := [][]float64{
		{1, 1},
		{-1, 1},
		{-1, -1},
		{1, -1},
	}
	expected := []float64{1, 1, 1, -1}
	p.Learn(dataset, expected)
	fmt.Println(p)
	fmt.Println(p.Activate([]float64{1, 1}))
	fmt.Println(p.Activate([]float64{-1, 1}))
	fmt.Println(p.Activate([]float64{-1, -1}))
	fmt.Println(p.Activate([]float64{1, -1}))
}
