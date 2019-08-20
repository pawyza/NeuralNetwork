package datasetPrep

import datasetPrep.graphics.Frame
import java.awt.Color
import java.awt.image.BufferedImage
import java.awt.image.BufferedImage.TYPE_BYTE_GRAY
import java.io.File
import javax.imageio.ImageIO

class Data(filePath : String){

    private var dataset : Array<Pair<Double,Array<Double>>> = arrayOf()

    init{
        dataset = CSVopener().loadFile(filePath)
        //dataset = CSVopener().loadFile(filePath).mapIndexed{ index, data -> Pair(data.first,Array(28){row -> Array(28){pixel -> data.second[row*28+pixel]}})}.toTypedArray()
        /*println(dataset.joinToString("\n\n") {"Value: ${it.first}\n" +
                it.second.joinToString("\n") { row -> row.joinToString(" ")}
        })
        println(CSVopener().loadFile(filePath).mapIndexed{ index, data -> Pair(data.first,Array(28){row -> Array(28){pixel -> data.second[row*28+pixel]}})}.toTypedArray().joinToString("\n\n") {"Value: ${it.first}\n" +
                    it.second.joinToString("\n") { row -> row.joinToString(" ")}})*/
    }


    fun displayData(index : Int, scale : Int){
        require(scale >= 1){"Scale must be positive number"}
        require(index<dataset.size && index>=0){"Index out of range"}
        val image = BufferedImage(28*scale,28*scale,TYPE_BYTE_GRAY)
        val graphic = image.graphics
        dataset[index].second.mapIndexed { index, value ->
            graphic.color = Color((value*255).toInt(),(value*255).toInt(),(value*255).toInt())
            graphic.fillRect((index%28)*scale,(index/28)*scale,scale,scale)
        }
        //ImageIO.write(image,"jpg",File("D:\\nnLearning\\testData\\MNISTCSV\\test.jpg"))
        Frame(image)
    }

    fun getData(i : Int) : Pair<Double,Array<Double>> = dataset[i]
    fun getDataSize() : Int = dataset.size
    fun getInputSize() : Int = dataset[0].second.size
}