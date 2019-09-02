package neuralNetworkBasics

import java.text.DecimalFormat
import kotlin.math.pow

/**
 * Class containing functions allowing user to create NN adjusted for users needs
 *
 * @property layers array containing number of neurons in every layer in order input->hidden->output
 * @property learningRate value which changes the speed of adjusting biases and weights by multiplying gradient, should be value from 0.0-1.0 for proper NN performance
 * @property weights array of Matrix objects containing weights for every neuron and layer
 * @property biases array of Double arrays containing biases for every neuron and layer
 * @property activations array of Double arrays containing activations for every neuron and layer
 */
class NeuralNetwork(val layers: Array<Int>, var learningRate : Double) {

    private var weights: Array<Matrix> = emptyArray()
    private var biases: Array<Array<Double>> = emptyArray()
    private var activations: Array<Array<Double>> = emptyArray()

    /**
     * Prepares NN by creating its structure
     */
    init {
        weights = NetworkLogic.createMatrixFromLayers(layers)
        biases = NetworkLogic.createBiasesArray(layers)
        activations = NetworkLogic.createActivationsArray(layers)

        val functionContainer: FunctionContainer = object : FunctionContainer {
            /**
             * Function which will determinate how biases and weights will be generated
             */
            override fun fillingFunction(): Double {
                return java.util.Random().nextGaussian() / layers[0].toDouble().pow(0.5)
            }
        }

        NetworkLogic.fillAllMatrices(weights, functionContainer)
        NetworkLogic.fillAllArrays(biases, functionContainer)
    }

    /**
     * Calculates activations for all neurons in network using biases, weights and activations of former layers
     *
     * @param input array for input data that will be feed into network in order to get predictions
     * @return array of last layer activations in other words output activations or just results
     */
    fun predict(input: Array<Double>): Array<Double> {
        require(activations[0].size == input.size) { "Input values and input layer must be the same size" }
        activations[0] = input
        for (layer in 0 until layers.size - 1) {
            activations[layer + 1] =
                NetworkLogic.calculateLayerOutput(weights[layer], biases[layer], activations[layer])
        }
        return activations.last()
    }

    /**
     * Feeds network with calculated gradients
     *
     * @param correctAnswer int value indicating which output neuron should have highest value activations
     */
    fun learnNetwork(correctAnswer: Int) {
        val gradientsW = calculateWeightGradients(correctAnswer)
        val gradientsB = calculateBiasesGradients(correctAnswer)

        for (layer in layers.size - 2 downTo 0) {
            for (neuron in 0 until layers[layer + 1]) {
                biases[layer][neuron] = biases[layer][neuron] - learningRate * gradientsB[layer][neuron]
            }
            for (neuron in 0 until layers[layer]) {
                for (outputNeuron in 0 until layers[layer + 1]) {
                    weights[layer].setValue(
                        neuron = neuron,
                        toNeuron = outputNeuron,
                        value = weights[layer].getValue(
                            neuron,
                            outputNeuron
                        ) - learningRate * activations[layer][neuron] * gradientsW[layer].getValue(neuron, outputNeuron)
                    )
                }
            }
        }
    }

    /**
     * Calculates total cost for all output neurons
     *
     * @param correctAnswer int value indicating which output neuron should have highest value activations
     * @return total cost for all output neurons
     */
    private fun totalCost(correctAnswer: Int): Double = activations.last().mapIndexed { neuron, result ->
        (when (neuron == correctAnswer) {
            true -> 1.0
            false -> 0.0
        } - result).pow(2)
    }.sum()

    /**
     * Calculates gradients required for weight adjusting
     *
     * @param correctAnswer int value indicating which output neuron should have highest value activations
     * @return array of Matrix objects containing gradients required for weight adjusting
     */
    private fun calculateWeightGradients(correctAnswer: Int): Array<Matrix> {
        val gradients = NetworkLogic.createMatrixFromLayers(layers)
        for (layer in (0 until layers.size - 1).reversed()) {
            for (inputNeuron in 0 until layers[layer]) {
                for (neuron in 0 until layers[layer + 1]) {
                    gradients[layer].setValue(
                        inputNeuron, neuron, when (layer == layers.size - 2) {
                            true -> (activations[layer + 1][neuron] * (1 - activations[layer + 1][neuron])) * 2 * (activations[layer + 1][neuron] - when (neuron == correctAnswer) {
                                true -> 1.0
                                false -> 0.0
                            })
                            false -> {
                                activations[layer + 1][neuron] * (1 - activations[layer + 1][neuron]) * weights[layer + 1].getColumnElements(
                                    neuron
                                ).mapIndexed { outputIdx, weight ->
                                    weight * gradients[layer + 1].getValue(
                                        neuron,
                                        outputIdx
                                    )
                                }.sum()
                            }
                        }
                    )
                }
            }
        }
        return gradients
    }

    /**
     * Calculates gradients required for biases adjusting
     *
     * @param correctAnswer int value indicating which output neuron should have highest value activations
     * @return array of Double arrays containing gradients required for biases adjusting
     */
    private fun calculateBiasesGradients(correctAnswer: Int): Array<Array<Double>> {
        val gradients = Array(activations.size - 1) { Array(activations[it + 1].size) { 0.0 } }
        for (layer in gradients.indices.reversed()) {
            for (neuron in gradients[layer].indices)
                when (layer == gradients.size - 1) {
                    true -> gradients[layer][neuron] =
                        activations[layer + 1][neuron] * (1 - activations[layer + 1][neuron]) * 2 * (activations[layer + 1][neuron] - when (neuron == correctAnswer) {
                            true -> 1.0
                            false -> 0.0
                        })
                    false -> {
                        gradients[layer][neuron] =
                            activations[layer + 1][neuron] * (1 - activations[layer + 1][neuron]) * weights[layer + 1].getColumnElements(
                                neuron
                            ).mapIndexed { outputIdx, weight -> weight * gradients[layer + 1][outputIdx] }.sum()

                    }
                }
        }
        return gradients
    }

    /**
     * Prints actual NN states including biases, weights, activations with respect to layers, neurons and output neurons
     */
    fun showNeuralNetworkState() {
        val formatter = DecimalFormat("0.00000")
        println("\nBiases")
        biases.mapIndexed { layerIndex, layer ->
            println(layer.mapIndexed { neuronIndex, bias ->
                "Layer: ${layerIndex + 1} Neuron: $neuronIndex Bias:${when (bias < 0) {
                    true -> ""; false -> " "
                } + formatter.format(bias)
                }"
            }.joinToString(" || "))
        }
        println("\nWeights")
        weights.mapIndexed { layerIdx, layer -> println("Layer: $layerIdx\n${layer.printMatrix()}") }
        println("\nValues")
        activations.mapIndexed { layerIdx, layer ->
            println(layer.mapIndexed { neuronIdx, x ->
                "Layer: $layerIdx Neuron: $neuronIdx Value:${when (x < 0) {
                    true -> ""; false -> " "
                } + formatter.format(x)
                }"
            }.joinToString(" || "))
        }
    }


    /**
     * Prints actual NN states including biases, weights, activations and gradients for given answer with respect to layers, neurons and output neurons
     *
     * @param correctAnswer int value indicating which output neuron should have highest value activations
     */
    fun showNeuralNetworkFullState(correctAnswer: Int) {
        showNeuralNetworkState()
        val formatter = DecimalFormat("0.00000")
        val gradientsW = calculateWeightGradients(correctAnswer)
        val gradientsB = calculateBiasesGradients(correctAnswer)
        println("\nGradients W")
        gradientsW.mapIndexed { layerIdx, layer -> println("Layer: $layerIdx\n${layer.printMatrix()}") }
        println("\nGradients B")
        gradientsB.mapIndexed { layerIdx, layer ->
            println(layer.mapIndexed { neuronIdx, e ->
                "Layer: $layerIdx Neuron: $neuronIdx Gradient B: ${when (e < 0) {
                    true -> ""; false -> " "
                } + formatter.format(e)
                }"
            }.joinToString("\n"))
        }
    }
}