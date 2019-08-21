package neuralNetworkBasics

import java.text.DecimalFormat

/**
 * Matrix of doubles implementation
 *
 * @property x number of elements in row
 * @property y number of rows
 * @property matrix array of arrays holding Matrix object values
 */
internal class Matrix(x: Int, y: Int) {

    private var matrix = emptyArray<Array<Double>>()

    init {
        require(x > 0 && y > 0) { "Size cannot be equal 0" }
        matrix = Array(y) { Array(x) { .0 } }
    }

    /**
     * Creates matrix of given size
     *
     * @param x number of elements in row
     * @param y number of rows
     * @return returns matrix of zeros
     */
    private fun createMatrix(x: Int, y: Int): Array<Array<Double>> = Array(y) { Array(x) { .0 } }

    /**
     * Change every element of matrix to given number
     *
     * @param number given value
     */
    internal fun fillWithNumber(number: Double) {
        matrix.mapIndexed() { r, row ->
            for (e in row.indices) {
                matrix[r][e] = number
            }
        }
    }

    /**
     * Change every element of matrix to given fillingFunction result
     *
     * @param fillingFunction given filling fillingFunction
     */
    internal fun fillWithFunction(functionContainer: FunctionContainer) {
        matrix.mapIndexed() { r, row ->
            for (e in row.indices) {
                matrix[r][e] = functionContainer.fillingFunction()
            }
        }
    }

    /**
     * Function that translates Matrix
     */
    internal fun translate() {
        matrix = Array(matrix[0].size) { row -> Array(matrix.size) { matrix[it][row] } }
    }

    /**
     *  Function that do multiply matrix with given array
     *
     *  @param arrayB given array
     *  @return result of matmul function
     */
    fun matmulWith(arrayB: Array<Double>): Array<Double> {
        require(arrayB.size == matrix[0].size) { "Cannot multiply matrix ${matrix.size}x${matrix[0].size} with ${arrayB.size}x1" }
        val result = Array(matrix.size) { 0.0 }
        for (row in matrix.indices) {
            for (element in matrix[0].indices) {
                result[row] += matrix[row][element] * arrayB[element]
            }
        }
        return result
    }

    /**
     * Creates string representing matrix content
     *
     * @return returns string representation of matrix
     */
    internal fun printMatrix(): String {
        val formatter = DecimalFormat("0.00000")
        return matrix.mapIndexed { outputNeuronIdx, row ->
            row.mapIndexed { inputNeuronIdx, weight ->
                "Input neuron: $inputNeuronIdx Output neuron: $outputNeuronIdx Weight: ${when (weight < 0) {
                    true -> ""; false -> " "
                } + formatter.format(weight)}"
            }.joinToString(" || ")
        }.joinToString("\n")
    }

    /**
     * Returns array of values of selected matrix column
     *
     * @param column index of selected column
     * @return array of column values
     */
    internal fun getColumnElements(column: Int): Array<Double> = Array(matrix.size) { matrix[it][column] }

    /**
     * Returns array of values of selected matrix row
     *
     * @param row index of selected row
     * @return array of row values
     */
    internal fun getRowElements(row: Int): Array<Double> = Array(matrix[0].size) { matrix[row][it] }

    /**
     * Changes value in Matrix for given coordinates, function parameters are named neuron and toNeuron for easier implementation in NN
     *
     * @param neuron column index of matrix
     * @param toNeuron row index of matrix
     * @param value value that will override old value
     */
    internal fun setValue(neuron: Int, toNeuron: Int, value: Double) {
        matrix[toNeuron][neuron] = value
    }

    /**
     * Returns value for given coordinates in Matrix, function parameters are named neuron and toNeuron for easier implementation in NN
     *
     * @param neuron column index of matrix
     * @param toNeuron row index of matrix
     * @return value of given coordinates
     */
    internal fun getValue(neuron: Int, toNeuron: Int): Double = matrix[toNeuron][neuron]
}
