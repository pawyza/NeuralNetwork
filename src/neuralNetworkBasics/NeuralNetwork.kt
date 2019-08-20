package neuralNetworkBasics

import java.text.DecimalFormat
import kotlin.math.exp
import kotlin.math.pow

class NeuralNetwork(val layers : Array<Int>,var learningRate : Double) {

    private var weightsMatrix : Array<WeightsMatrix> = emptyArray()
    private var biases : Array<Array<Double>> = emptyArray()
    private var a : Array<Array<Double>> = emptyArray()

    init {
        weightsMatrix = NetworkLogic.createMatrixFromLayers(layers)//2,3   3,5   5,2
        biases = NetworkLogic.createBiasesFromLayers(layers)
        a = Array(layers.size){ Array(layers[it]){0.0}}

        val functionContainer: FunctionContainer = object : FunctionContainer {
            /**
             * Function is rule which will generate new matrix values
             */
            override fun fillingFunction(): Double {
                return java.util.Random().nextGaussian()/layers[0].toDouble().pow(0.5)
            }
        }

        NetworkLogic.fillAllMatrix(weightsMatrix, functionContainer)
        NetworkLogic.fillAllArrays(biases, functionContainer)
    }

    fun predict(input : Array<Double>) : Array<Double>{
        require(a[0].size == input.size){"Input values and input layer must be the same size"}
        a[0] = input
        for(layer in 0 until layers.size-1){
            val matrix = weightsMatrix[layer]
            a[layer + 1] = NetworkLogic.calculateLayerOutput(weightsMatrix[layer], biases[layer], a[layer])
        }
        return a.last()
    }

    private fun activationFunction(x : Double) : Double = 1/(1+ exp(-x))

    fun showNeuralNetworkState() {
        val formatter = DecimalFormat("0.00000")
        println("\nBiases")
        biases.mapIndexed {layerIndex ,layer -> println(layer.mapIndexed {neuronIndex, bias -> "Layer: ${layerIndex+1} Neuron: $neuronIndex Bias:${when (bias < 0) {
                true -> ""; false -> " "
            } + formatter.format(bias)
        }"}.joinToString(" || "))}
        println("\nWeights")
        weightsMatrix.mapIndexed{layerIndex, layer -> println("Layer: $layerIndex\n${layer.printMatrix()}")}
        println("\nValues")
        a.mapIndexed {layerIndex ,layer -> println(layer.mapIndexed {neuronIndex, x -> "Layer: $layerIndex Neuron: $neuronIndex Value:${when (x < 0) {
            true -> ""; false -> " "
        } + formatter.format(x)
        }"}.joinToString(" || "))}
        //weightsMatrix.mapIndexed{ind, matrix -> matrix.zip(biases[ind]); println(matrix.printMatrix()); println() }
    }

    fun showNeuralNetworkFullState(correctAnswer: Int) {
        showNeuralNetworkState()
        val formatter = DecimalFormat("0.00000")
        val gradientsW = allWGradients(correctAnswer)
        val gradientsB = allBGradients(gradientsW,correctAnswer)
        /*println("\nError")
        errors.mapIndexed {layerIndex ,layer -> println(layer.mapIndexed {neuronIndex, e -> "Layer: $layerIndex Neuron: $neuronIndex Error: ${when (e < 0) {
            true -> ""; false -> " "
        } + formatter.format(e)
        }"}.joinToString("\n"))}*/
        println("\nGradients W")
        gradientsW.mapIndexed {layerIdx ,layer -> layer.matrix.mapIndexed{neuronIdx, neuron -> neuron.mapIndexed{outputIdx, weight -> println("Layer: $layerIdx Neuron: $neuronIdx Gradient W: ${weight*a[layerIdx][neuronIdx]}")}}}
        println("\nGradients B")
        gradientsB.mapIndexed {layerIdx ,layer -> println(layer.mapIndexed {neuronIdx, e -> "Layer: $layerIdx Neuron: $neuronIdx Gradient B: ${when (e < 0) {
            true -> ""; false -> " "
        } + formatter.format(e)
        }"}.joinToString("\n"))}
    }

    fun calculateFullCost(results : Array<Double>, correctAnswer : Int) : Double = results.mapIndexed{index, result -> (result- when(index==correctAnswer){
        true -> 1.0
        false -> 0.0
    }).pow(2)}.sum()/results.size

    private fun totalError(correctAnswer: Int) : Double = a.last().mapIndexed {neuron, result -> (when(neuron == correctAnswer){
                        true -> 1.0
                        false -> 0.0
                    } - result).pow(2)
                }.sum()

    private fun allWGradients(correctAnswer: Int) : Array<WeightsMatrix> {
        val gradients  = NetworkLogic.createMatrixFromLayers(layers)
        //weightsMatrix.map { println(it.printMatrix());println() }
        //gradients.map {println(it.printMatrix());println() }
        for(layer in (0 until layers.size-1).reversed()){
            for (inputNeuron in 0 until layers[layer]) {
                for (neuron in 0 until layers[layer+1]) {
                    //weightsMatrix.map { println(it.printMatrix());println() }
                    //gradients.map {println(it.printMatrix());println() }
                    gradients[layer].setWeight(inputNeuron,neuron, when (layer == layers.size - 2) {
                        true -> (a[layer+1][neuron] * (1 - a[layer+1][neuron])) * 2 * (a[layer+1][neuron] - when (neuron == correctAnswer) {
                            true -> 1.0
                            false -> 0.0
                        })
                        false -> {
                            //println( "${weightsMatrix[layer].getWeight(inputNeuron,neuron)} ${gradients[layer+1].getWeight(inputNeuron,neuron)}");
                            a[layer+1][neuron] * (1 - a[layer+1][neuron]) * weightsMatrix[layer+1].getColumnElements(neuron).mapIndexed{outputIdx, weight -> weight * gradients[layer+1].getWeight(neuron,outputIdx)}.sum()
                        }})
                    //gradients.map {println(it.printMatrix());println() }
                    }
                }
            }
        //gradients.map { it.map{ ele -> println(ele)}; println() }
        return gradients
    }

    private fun allBGradients(gradientsW : Array<WeightsMatrix>,correctAnswer: Int) : Array<Array<Double>> {
        val totalError = totalError(correctAnswer)
        val gradients = Array(a.size-1){Array(a[it+1].size){0.0}}
        for(layer in gradients.indices.reversed()){
            for(neuron in gradients[layer].indices)
                when(layer == gradients.size-1){
                    true -> gradients[layer][neuron] =  a[layer+1][neuron] * (1- a[layer+1][neuron]) * 2 * ( a[layer+1][neuron] - when(neuron ==correctAnswer){
                        true -> 1.0
                        false -> 0.0
                    })
                    false -> {
                        gradients[layer][neuron] = a[layer+1][neuron] * (1 - a[layer+1][neuron]) * weightsMatrix[layer+1].getColumnElements(neuron).mapIndexed{outputIdx, weight -> weight * gradients[layer+1][outputIdx]}.sum()

                    }
                }
        }
        //gradients.map { it.map{ ele -> println(ele)}; println() }
        return gradients
    }

    /*
    private fun allWGradients(result : Array<Double>, correctAnswer: Int) : Array<Array<Double>> {
        val weightErrors = Array(a.size){Array(a[it].size){0.0}}
        for(layer in weightErrors.indices.reversed()){
            for(neuron in weightErrors[layer].indices)
                when(layer == weightErrors.size-1){
                    true -> weightErrors[layer][neuron] = when(neuron ==correctAnswer){
                        true -> 1.0
                        false -> 0.0
                    } - result[neuron]
                    false -> {
                        val neuronWeightsSum = weightsMatrix[layer].getColumnElements(neuron).sum()
                        weightErrors[layer][neuron] =
                            weightsMatrix[layer].getColumnElements(neuron).mapIndexed {outputNeuron, weight -> weight/neuronWeightsSum*weightErrors[layer+1][outputNeuron]
                        }.sum()
                    }
                }
        }
        //weightErrors.map { it.map{ ele -> println(ele)}; println() }
        return weightErrors
    }
     */

    fun learnNetwork(correctAnswer: Int) {
        val gradientsW = allWGradients(correctAnswer)
        val gradientsB = allBGradients(gradientsW,correctAnswer)

        for(layer in layers.size-2 downTo 0) {
            for(neuron in 0 until layers[layer+1]) {
                biases[layer][neuron] = biases[layer][neuron] - learningRate * gradientsB[layer][neuron]
            }
            for (neuron in 0 until layers[layer]) {
                for (outputNeuron in 0 until layers[layer+1]){
                    weightsMatrix[layer].setWeight(neuron = neuron, toNeuron = outputNeuron, value = weightsMatrix[layer].getWeight(neuron,outputNeuron) - learningRate * a[layer][neuron] * gradientsW[layer].getWeight(neuron,outputNeuron))
                }
            }
        }

    }
}