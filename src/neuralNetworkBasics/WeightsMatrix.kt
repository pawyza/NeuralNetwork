package neuralNetworkBasics

import java.text.DecimalFormat

/**
 * WeightsMatrix of doubles implementation
 *
 * @property x number of elements in row
 * @property y number of rows
 */
internal class WeightsMatrix(x : Int, y : Int){

    var matrix = emptyArray<Array<Double>>()

    init{
        require(x>0 && y>0){"Size cannot be equal 0"}
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
    internal fun fillWithNumber(number : Double){
        matrix.mapIndexed() { r, row -> for(e in row.indices){matrix[r][e] = number} }
    }

    /**
     * Change every element of matrix to given fillingFunction result
     *
     * @param fillingFunction given filling fillingFunction
     */
    internal fun fillWithFunction(functionContainer : FunctionContainer){
        matrix.mapIndexed() { r, row -> for(e in row.indices){matrix[r][e] = functionContainer.fillingFunction()} }
    }

    /**
     * Change every element of matrix to given fillingFunction result
     *
     * @param fillingFunction given filling fillingFunction
     */
    internal fun translate(){
        matrix = Array(matrix[0].size){row -> Array(matrix.size){matrix[it][row]}}
    }

    /*
    fun zip(otherMatrix : WeightsMatrix){
        require(matrix.size == otherMatrix.matrix.size){"WeightsMatrix size must be equal"}
        var newMatrix = Array(matrix.size)  { Array(matrix[0].size+otherMatrix.matrix[0].size){ .0 } }
        newMatrix.mapIndexed{ ind, _ -> newMatrix[ind] = matrix[ind].plus(otherMatrix.matrix[ind])}
        matrix = newMatrix
    }
*/

    /**
     *  Function that do matmul function for matrix of this WeightsMatrix and given array
     *
     *  @param arrayB given array
     *  @return result of matmul function
     */
    fun matmulWith(arrayB : Array<Double>) : Array<Double>{
        require(arrayB.size == matrix[0].size){"Connot multiply matrix ${matrix.size}x${matrix[0].size} with ${arrayB.size}x1"}
        val result = Array(matrix.size){0.0}
        for(row in matrix.indices){
            for(element in matrix[0].indices){
                //println("${matrix[row][element]} : ${arrayB[element]}")
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
    internal fun printMatrix() : String{
        val formatter = DecimalFormat("0.00000")
        return matrix.mapIndexed {outputNeuronIdx, row -> row.mapIndexed {inputNeuronIdx, weight ->  "Input neuron: $inputNeuronIdx Output neuron: $outputNeuronIdx Weight: ${when(weight < 0){true -> ""; false -> " "} + formatter.format(weight)}"}.joinToString(" || ")}.joinToString("\n")
    }

    internal fun adjustWeightsForNeuron(neuron : Int ,gradient : Double){
        for(row in matrix.indices){
            matrix[row][neuron] =  matrix[row][neuron] * gradient
        }
    }

    internal fun getColumnElements(column : Int) : Array<Double> = Array(matrix.size){matrix[it][column]}
    internal fun getRowElements(row : Int) : Array<Double> = Array(matrix[0].size){matrix[row][it]}
    internal fun getSize() : Int = matrix.size*matrix[0].size
    internal fun setWeight(neuron : Int, toNeuron : Int, value : Double){matrix[toNeuron][neuron] = value}
    internal fun getWeight(neuron : Int, toNeuron : Int) : Double = matrix[toNeuron][neuron]
}
