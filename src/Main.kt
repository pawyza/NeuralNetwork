import datasetPrep.Data
import neuralNetworkBasics.NeuralNetwork
import java.text.DecimalFormat
import java.util.*

fun main(args: Array<String>) {

    println("Loading data...")
    val data = Data("D:\\nnLearning\\testData\\MNISTCSV\\mnist_train.csv")
    val testData = Data("D:\\nnLearning\\testData\\MNISTCSV\\mnist_test.csv")
    val layers = arrayOf(data.getInputSize(), 15, 10)
    val network = NeuralNetwork(layers, 0.1)
    testNetwork(network, data, testData)
/*
    val layers = arrayOf(2, 2, 2)
    val network = NeuralNetwork(layers,0.01)
    network.predict(arrayOf(1.0,0.0))
    network.learnNetwork(0)
    network.showNeuralNetworkFullState(0)
    for (i in 0..10000){
        network.predict(arrayOf(1.0,0.0))
        network.learnNetwork(0)
        network.predict(arrayOf(0.0,1.0))
        network.learnNetwork(1)
    }
    network.predict(arrayOf(1.0,0.0))
    network.learnNetwork(0)
    network.showNeuralNetworkFullState(0)
    network.predict(arrayOf(0.0,1.0))
    network.learnNetwork(1)
    network.showNeuralNetworkFullState(1)
*/
}

private fun testNetwork(network: NeuralNetwork, data: Data, testData: Data) {
    var correct = 0.0
    for (i in 0 until testData.getDataSize()) {
        val result = network.predict(testData.getData(i).second)
        when (result.indexOf(result.max()) == testData.getData(i).first.toInt()) {
            true -> correct++
        }
    }
    val untrained = (correct / testData.getDataSize()) * 100
    println("Correctness of untrained NN: ${DecimalFormat("#0.00").format(untrained)}%")
    val loops = 200
    val dataPackSize = 10
    val dataLoop = 10
    for (l in 0 until loops) {
        //val learningPack = mutableListOf<Pair<Array<Double>,Int>>()
        val dataPack = Array(dataPackSize) { Random().nextInt(data.getDataSize()) }
        for (r in 0 until dataLoop) {
            val tik = System.currentTimeMillis()
            correct = 0.0
            for (i in 0 until dataPackSize) {
                val result = network.predict(data.getData(dataPack[i]).second)
                when (result.indexOf(result.max()) == data.getData(dataPack[i]).first.toInt()) {
                    true -> correct++
                }
                network.learnNetwork(data.getData(dataPack[i]).first.toInt())
            }
            val tok = System.currentTimeMillis()
            if (l % when (loops * 0.01.toInt() == 0) {
                    true -> 1;false -> loops * 0.01.toInt()
                } == 0
            ) {
                println("Done ${DecimalFormat("#0.00").format((l * dataLoop + r) * 100 / (loops * dataLoop))}%")
                println("Correctness of NN: ${DecimalFormat("#0.00").format((correct / dataPackSize) * 100)}%")
                println(
                    "Estimated time left: ${DecimalFormat("###0").format(((loops * dataLoop) - (l * dataLoop + r)) * -(tik - tok) / 60000)} minutes ${DecimalFormat(
                        "#0"
                    ).format((((loops * dataLoop) - (l * dataLoop + r)) * -(tik - tok) % 60000) / 1000)} seconds"
                )
            }
        }
    }
    correct = 0.0
    for (i in 0 until testData.getDataSize()) {
        val result = network.predict(testData.getData(i).second)
        when (result.indexOf(result.max()) == testData.getData(i).first.toInt()) {
            true -> correct++
        }
    }
    println("Correctness of untrained NN: ${DecimalFormat("#0.00").format(untrained)}%")
    println("Correctness of trained NN: ${DecimalFormat("#0.00").format((correct / testData.getDataSize()) * 100)}%")
}