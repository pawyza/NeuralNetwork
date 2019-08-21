package neuralNetworkBasics

import kotlin.math.exp

/**
 * Object that contains functions necessary for NN to work that contains operations on NN properties
 */
internal object NetworkLogic {

    /**
     * Creates array of Matrix objects
     *
     * @param layers array containing number of neurons in every layer
     * @return array containing weights organized in matrix form
     */
    internal fun createMatrixFromLayers(layers: Array<Int>): Array<Matrix> {
        require(layers.size > 2) { "Network must contain more than 2 layers!" }
        return Array(layers.size - 1) { Matrix(layers[it], layers[it + 1]) }
    }

    /**
     * Creates array of biases
     *
     * @param layers array containing number of neurons in every layer
     * @return array containing arrays of biases for all layers
     */
    internal fun createBiasesArray(layers: Array<Int>): Array<Array<Double>> {
        require(layers.size > 2) { "Network must contain more than 2 layers!" }
        return Array(layers.size - 1) { Array(layers[it + 1]) { 0.0 } }
    }

    /**
     * Creates array of activations
     *
     * @param layers array containing number of neurons in every layer
     * @return array containing arrays of activations for all layers
     */
    internal fun createActivationsArray(layers: Array<Int>): Array<Array<Double>> {
        require(layers.size > 2) { "Network must contain more than 2 layers!" }
        return Array(layers.size) { Array(layers[it]) { 0.0 } }
    }

    /**
     * Fills all arrays in given array with given value
     *
     * @param array array that should be filled
     * @param value value that will replace current values
     */
    internal fun fillAllArrays(array: Array<Array<Double>>, value: Double) {
        array.map { biasesArray ->
            for (element in biasesArray.indices) {
                biasesArray[element] = value
            }
        }
    }

    /**
     * Fills all arrays in given array with given function
     *
     * @param array array that should be filled
     * @param functionContainer contains rules that will generate new values
     */
    internal fun fillAllArrays(array: Array<Array<Double>>, functionContainer: FunctionContainer) {
        array.map { biasesArray ->
            for (element in biasesArray.indices) {
                biasesArray[element] = functionContainer.fillingFunction()
            }
        }
    }

    /**
     * Call Matrix filling method
     *
     * @param matrices contains all weights matrices
     * @param value value that will replace current values
     */
    internal fun fillAllMatrices(matrices: Array<Matrix>, value: Double) {
        matrices.map { it.fillWithNumber(value) }
    }

    /**
     * Call Matrix filling method
     *
     * @param matrices contains all weights matrices
     * @param functionContainer contains rules that will generate new values
     */
    internal fun fillAllMatrices(matrices: Array<Matrix>, functionContainer: FunctionContainer) {
        matrices.map { it.fillWithFunction(functionContainer) }
    }

    /**
     * Calculates output of layer
     * @param weightMatrix layers weights
     * @param biases layers biases
     * @param input previous layer activations
     * @return array of calculated layers activations values
     */
    internal fun calculateLayerOutput(
        weightMatrix: Matrix,
        biases: Array<Double>,
        input: Array<Double>
    ): Array<Double> = weightMatrix.matmulWith(input)
        .mapIndexed { i, result -> result + biases[i] }
        .map { activationFunction(it) }.toTypedArray()

    /**
     * Sigmoid function
     *
     * @param x sigmoid function parameter
     * @return f(x) result
     */
    private fun activationFunction(x: Double): Double = 1 / (1 + exp(-x))
}