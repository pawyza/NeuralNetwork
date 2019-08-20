package datasetPrep.graphics

import java.awt.Dimension
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.image.BufferedImage
import javax.swing.JButton
import javax.swing.JFrame

class Frame(image : BufferedImage) : JFrame("Image"){

    init {
        add(Panel(image))
        defaultCloseOperation = JFrame.DISPOSE_ON_CLOSE
        pack()
        isVisible = true
    }
}