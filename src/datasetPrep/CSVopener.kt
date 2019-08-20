package datasetPrep

import java.io.File
import java.util.*

internal class CSVopener(){

    fun loadFile(filePath : String) : Array<Pair<Double,Array<Double>>>{
        val scanner = Scanner(File(filePath))
        var result : MutableList<Pair<Double, Array<Double>>> = mutableListOf()
        while (scanner.hasNext()) {
            var data = scanner.next().split(",").toMutableList()
            result.add(Pair(data.removeAt(0).toDouble(),data.map { it.toDouble()/255 }.toTypedArray()))
        }
        //prDouble("V: ${result[0].first} Pixels: ${result[0].second.joDoubleoString(" ")}")
        return result.toTypedArray()
    }
}