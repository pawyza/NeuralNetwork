package datasetPrep.graphics

import java.awt.Dimension
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.image.BufferedImage
import javax.swing.JPanel

class Panel(private val image : BufferedImage) : JPanel(){

    init {
        preferredSize = Dimension(image.width,image.height)
    }

    override fun paintComponent(g: Graphics?) {
        val g2d = g as Graphics2D
        g2d.drawImage(image,0,0,this)
    }
}