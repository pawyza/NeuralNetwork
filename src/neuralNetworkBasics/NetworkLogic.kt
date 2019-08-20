package neuralNetworkBasics

import kotlin.math.exp

internal object NetworkLogic {

    /**
     * Creates array of WeightsMatrix objects
     *
     * @param layers array containing number of neurons in every layer
     * @return array containing weights organized in matrix form
     */
    internal fun createMatrixFromLayers(layers : Array<Int>) : Array<WeightsMatrix>{
        require(layers.size > 2){"Network must contain more than 2 layers!"}
        return Array(layers.size-1){WeightsMatrix(layers[it],layers[it+1])}
    }

    /**
     * Creates array of biases
     *
     * @param layers array containing number of neurons in every layer
     * @return array containing array of biases for all layers
     */
    internal fun createBiasesFromLayers(layers : Array<Int>) : Array<Array<Double>>{
        require(layers.size > 2){"Network must contain more than 2 layers!"}
        return Array(layers.size-1){Array(layers[it+1]){0.0}}
    }

    internal fun fillAllArrays(array : Array<Array<Double>>,functionContainer: FunctionContainer){
        array.map{ biasesArray -> for(element in biasesArray.indices){biasesArray[element] = functionContainer.fillingFunction()}}
    }

    /**
     * Call WeightsMatrix filling method
     *
     * @param matrix contains all weights matrix
     * @param number value that will replace current values
     */
    internal fun fillAllMatrix(matrix : Array<WeightsMatrix>, number : Double){
        matrix.map{it.fillWithNumber(number)}
    }

    internal fun calculateLayerOutput(matrix : WeightsMatrix, biases : Array<Double>, input : Array<Double>) : Array<Double> = matrix.matmulWith(input)
        .mapIndexed { i, result ->result + biases[i] }
        .map {activationFunction(it) }.toTypedArray()

    private fun activationFunction(x : Double) : Double = 1/(1+ exp(-x))

    /**
     * Call WeightsMatrix filling method
     *
     * @param matrix contains all weights matrix
     * @param functionContainer contains rules that will generate new values
     */
    internal fun fillAllMatrix(matrix : Array<WeightsMatrix>, functionContainer : FunctionContainer){
        matrix.map{it.fillWithFunction(functionContainer)}
    }
}