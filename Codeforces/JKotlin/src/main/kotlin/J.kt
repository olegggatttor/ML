import java.lang.Integer.min
import kotlin.math.max

class Matrix3D(val depth: Int, val height: Int, val width: Int, val values: Array<Array<FloatArray>>) {

    constructor(d: Int, h: Int, w: Int) : this(d, h, w, Array(d) { Array(h) { FloatArray(w) } })

    operator fun get(d: Int, i: Int, j: Int): Float {
        return this.values[d][i][j]
    }

    operator fun set(d: Int, i: Int, j: Int, value: Float) {
        this.values[d][i][j] = value
    }

    inline fun apply(f: (Int, Int, Int) -> Float): Matrix3D {
        val result = Matrix3D(depth, height, width)
        for (i in 0 until depth) {
            for (j in 0 until height) {
                for (k in 0 until width) {
                    result[i, j, k] = f.invoke(i, j, k)
                }
            }
        }
        return result
    }

    companion object {
        fun read3DMx(n: Int, d: Int, vals: FloatArray): Matrix3D {
            val res = Array(d) { Array(n) { FloatArray(n) } }
            for (k in 0 until d) {
                for (i in 0 until n) {
                    for (j in 0 until n) {
                        res[k][i][j] = vals[k * n * n + i * n + j]
                    }
                }
            }
            return Matrix3D(d, n, n, res)
        }
    }
}

class Matrix4D(
    val h: Int,
    val depth: Int,
    val height: Int,
    val width: Int,
    val values: Array<Array<Array<FloatArray>>>
) {

    constructor(h: Int, d: Int, k: Int) : this(h, d, k, k, Array(h) { Array(d) { Array(k) { FloatArray(k) } } })


    operator fun get(h: Int, d: Int, i: Int, j: Int): Float {
        return this.values[h][d][i][j]
    }

    operator fun set(h: Int, d: Int, i: Int, j: Int, value: Float) {
        values[h][d][i][j] = value
    }

    companion object {
        fun read4DMx(h: Int, d: Int, k: Int, values: FloatArray): Matrix4D {
            val res = Array(h) { Array(d) { Array(k) { FloatArray(k) } } }
            for (hh in 0 until h) {
                for (dd in 0 until d) {
                    for (i in 0 until k) {
                        for (j in 0 until k) {
                            res[hh][dd][i][j] = values[hh * d * k * k + dd * k * k + i * k + j]
                        }
                    }
                }
            }
            return Matrix4D(h, d, k, k, res)
        }
    }
}

open class Computation(val inputMatrix: Matrix3D) {
    lateinit var dx: Matrix3D
    lateinit var outputMatrix: Matrix3D

    open fun compute() {

    }

    open fun derivative(prev: Computation) {

    }
}

class Relu(_inputMatrix: Matrix3D, val alpha: Float) : Computation(_inputMatrix) {

    override fun compute() {
        outputMatrix = inputMatrix.apply { i, j, k ->
            when (inputMatrix[i, j, k] < 0) {
                true -> inputMatrix[i, j, k] / alpha
                false -> inputMatrix[i, j, k]
            }
        }
    }

    override fun derivative(prev: Computation) {
        prev.dx = dx.apply { i, j, k ->
            when (prev.outputMatrix[i, j, k] < 0) {
                true -> dx[i, j, k] / alpha
                false -> dx[i, j, k]
            }
        }
    }
}

class Bias(_inputMatrix: Matrix3D, val bs: FloatArray) : Computation(_inputMatrix) {
    val dBs = FloatArray(bs.size)

    override fun compute() {
        outputMatrix = inputMatrix.apply { i, j, k ->
            inputMatrix[i, j, k] + bs[i]
        }
    }

    override fun derivative(prev: Computation) {
        prev.dx = dx
        // dBiases?
        for (k in 0 until dx.depth) {
            for (i in 0 until dx.height) {
                for (j in 0 until dx.width) {
                    dBs[k] += dx[k, i, j]
                }
            }
        }
    }
}

class Pool(_inputMatrix: Matrix3D, val s: Int) : Computation(_inputMatrix) {

    override fun compute() {
        outputMatrix = Matrix3D(inputMatrix.depth, inputMatrix.height / min(s, inputMatrix.height), inputMatrix.width / min(s, inputMatrix.width))
        for (i in 0 until outputMatrix.depth) {
            for (j in 0 until outputMatrix.height) {
                for (k in 0 until outputMatrix.width) {
                    outputMatrix[i, j, k] = sliceMax(s * j, s * (j + 1), s * k, s * (k + 1), i)
                }
            }
        }
    }

    private fun sliceMax(jStart: Int, jEnd: Int, kStart: Int, kEnd: Int, d: Int): Float {
        var maximum = -Float.MAX_VALUE
        for (j in jStart until jEnd) {
            for (k in kStart until kEnd) {
                maximum = max(maximum, inputMatrix[d, j, k])
            }
        }
        return maximum
    }

    override fun derivative(prev: Computation) {
        prev.dx = Matrix3D(prev.outputMatrix.depth, prev.outputMatrix.height, prev.outputMatrix.width)
        for (i in 0 until outputMatrix.depth) {
            for (j in 0 until outputMatrix.height) {
                for (k in 0 until outputMatrix.width) {
                    val maxInSquare = outputMatrix[i, j, k]
                    for (p in s * j until s * (j + 1)) {
                        for (q in s * k until s * (k + 1)) {
                            prev.dx[i, p, q] = when (prev.outputMatrix[i, p, q] == maxInSquare) {
                                true -> dx[i, j, k]
                                false -> 0.0f
                            }
                        }
                    }
                }
            }
        }
    }
}

class Cnv(
    _inputMatrix: Matrix3D, val padType: String, val h: Int,
    val k: Int, val s: Int, var p: Int, val wMatrix: Matrix4D
) : Computation(_inputMatrix) {
    val dWMatrix = Matrix4D(h, wMatrix.depth, k)

    override fun compute() {
        val cnvResSize = (inputMatrix.values[0].size + 2 * p - k) / s + 1
        outputMatrix = Matrix3D(h, cnvResSize, cnvResSize)
        val paddedInput = when (padType) {
            "cnvm" -> mirrorPad(inputMatrix)
            "cnve" -> extendPad(inputMatrix)
            else -> cyclicPad(inputMatrix)
        }
        for (hh in 0 until h) {
            for (i in 0 until cnvResSize) {
                for (j in 0 until cnvResSize) {
                    for (zDepth in 0 until inputMatrix.depth) {
                        for (p in 0 until k) {
                            for (q in 0 until k) {
                                outputMatrix[hh, i, j] += wMatrix[hh, zDepth, p, q] * paddedInput[zDepth, p + s * i, q + s * j]
                            }
                        }
                    }
                }
            }
        }
    }

    override fun derivative(prev: Computation) {
        val paddedRes = cyclicPad(Matrix3D(prev.outputMatrix.depth, prev.outputMatrix.height, prev.outputMatrix.width))
        val paddedPrev = when (padType) {
            "cnvm" -> mirrorPad(prev.outputMatrix)
            "cnve" -> extendPad(prev.outputMatrix)
            else -> cyclicPad(prev.outputMatrix)
        }
        for (hh in 0 until h) {
            for (i in 0 until dx.height) {
                for (j in 0 until dx.width) {
                    for (z in 0 until prev.outputMatrix.depth) {
                        for (y in 0 until k) {
                            for (x in 0 until k) {
                                dWMatrix[hh, z, y, x] += paddedPrev[z, s * i + y, s * j + x] * dx[hh, i, j]
                                paddedRes[z, s * i + y, s * j + x] += wMatrix[hh, z, y, x] * dx[hh, i, j]

                            }
                        }
                    }
                }
            }
        }
        prev.dx = when (padType) {
            "cnvm" -> rmMirrorPad(paddedRes)
            "cnve" -> rmExtendPad(paddedRes)
            else -> rmCyclicPad(paddedRes)
        }
    }

    fun rmCyclicPad(mx : Matrix3D) : Matrix3D {
        val res = Matrix3D(mx.depth, mx.height - 2 * p, mx.width - 2 * p)
        for (d in 0 until mx.depth) {
            for (i in 0 until mx.height) {
                val ix = when {
                    i < p -> i + (res.height - p)
                    i + p < mx.height -> i - p
                    else -> i + p - mx.height
                }
                for (j in 0 until mx.width) {
                    val jx = when {
                        j < p -> j + (res.width - p)
                        j + p < mx.width -> j - p
                        else -> j + p - mx.width
                    }
                    res[d, ix, jx] += mx[d, i, j]
                }
            }
        }
        return res
    }

    fun rmExtendPad(mx: Matrix3D): Matrix3D {
        val res = Matrix3D(mx.depth, mx.height - 2 * p, mx.width - 2 * p)
        for (d in 0 until mx.depth) {
            for (i in 0 until mx.height) {
                val ix = when {
                    i < p -> 0
                    i + p < mx.height -> i - p
                    else -> res.height - 1
                }
                for (j in 0 until mx.width) {
                    val jx = when {
                        j < p -> 0
                        j + p < mx.width -> j - p
                        else -> res.width - 1
                    }
                    res[d, ix, jx] += mx[d, i, j]
                }
            }
        }
        return res
    }

    fun rmMirrorPad(mx: Matrix3D): Matrix3D {
        val res = Matrix3D(mx.depth, mx.height - 2 * p, mx.width - 2 * p)
        for (d in 0 until mx.depth) {
            for (i in 0 until mx.height) {
                val ix = when {
                    i < p -> p - i
                    i + p < mx.height -> i - p
                    else -> res.height - p + (mx.height - i) - 2
                }
                for (j in 0 until mx.width) {
                    val jx = when {
                        j < p -> p - j
                        j + p < mx.width -> j - p
                        else -> res.width - p + (mx.width - j) - 2
                    }
                    res[d, ix, jx] += mx[d, i, j]
                }
            }
        }
        return res
    }

    private fun mirrorPad(mx: Matrix3D): Matrix3D {
        val res = Matrix3D(mx.depth, mx.height + 2 * p, mx.width + 2 * p)
        // fillCenter
        for (d in 0 until res.depth) {
            for (i in p until res.height - p) {
                for (j in p until res.width - p) {
                    res[d, i, j] = mx[d, i - p, j - p]
                }
            }
        }
        for (d in 0 until res.depth) {
            for (i in 0 until res.height) {
                for (j in 0 until res.width) {
                    res[d, i, j] = mx[d, getMirrorIndex(i, res.height), getMirrorIndex(j, res.height)]
                }
            }
        }
        return res
    }

    private fun extendPad(mx: Matrix3D): Matrix3D {
        val res = Matrix3D(mx.depth, mx.height + 2 * p, mx.width + 2 * p)
        // fillCenter
        for (d in 0 until res.depth) {
            for (i in p until res.height - p) {
                for (j in p until res.width - p) {
                    res[d, i, j] = mx[d, i - p, j - p]
                }
            }
        }
        for (d in 0 until res.depth) {
            // fill vertically
            for (i in 0 until p) {
                for (j in p until res.width - p) {
                    res[d, i, j] = mx[d, 0, j - p]
                    res[d, res.height - i - 1, j] = mx[d, mx.height - 1, j - p]
                }
            }
            // fill horizontally
            for (i in 0 until res.height) {
                for (j in 0 until p) {
                    res[d, i, j] = res[d, i, p]
                    res[d, i, res.width - j - 1] = res[d, i, res.width - p - 1]
                }
            }
        }
        return res
    }

    private fun cyclicPad(mx: Matrix3D): Matrix3D {
        val res = Matrix3D(mx.depth, mx.height + 2 * p, mx.width + 2 * p)
        // fillCenter
        for (d in 0 until res.depth) {
            for (i in p until res.height - p) {
                for (j in p until res.width - p) {
                    res[d, i, j] = mx[d, i - p, j - p]
                }
            }
        }
        for (d in 0 until res.depth) {
            // fill vertically
            for (i in 0 until p) {
                for (j in p until res.width - p) {
                    res[d, i, j] = mx[d, Math.floorMod(mx.height - p + i, mx.height), j - p]
                    res[d, res.height - i - 1, j] = mx[d, Math.floorMod(p - i - 1, mx.height), j - p]
                }
            }
            // fill horizontally
            for (i in 0 until res.height) {
                for (j in 0 until p) {
                    res[d, i, j] = res[d, i, Math.floorMod(mx.width - p + j, mx.width) + p]
                    res[d, i, res.width - j - 1] = res[d, i,Math.floorMod(p - j - 1, mx.width) + p]
                }
            }
        }
        return res
    }

    private fun getMirrorIndex(index: Int, l: Int): Int {
        return when {
            index < p -> p - index
            index < l - p -> index - p
            else -> (l - 2 * p) - (p - l + index) - 2
        }
    }
}

class Init(_ip: Matrix3D) : Computation(_ip) {
    override fun compute() {
        outputMatrix = inputMatrix
    }
}

fun main() {
    var info = readLine()!!.split(" ").map { it.toInt() }.toIntArray()
    val N = info[0]
    val D = info[1]
    info = info.sliceArray(2..info.lastIndex)
    var mx = Matrix3D.read3DMx(N, D, info.map { it.toFloat() }.toFloatArray())
    val amountOfActions = readLine()!!.toInt()
    val computations = ArrayList<Computation>()
    computations.add(Init(mx))
    computations[0].compute()
    repeat(amountOfActions) {
        val action = readLine()!!.split(" ")
        when (action[0]) {
            "relu" -> {
                val cur = Relu(mx, action[1].toFloat())
                cur.compute()
                mx = cur.outputMatrix
                computations.add(cur)
            }
            "pool" -> {
                val cur = Pool(mx, action[1].toInt())
                cur.compute()
                mx = cur.outputMatrix
                computations.add(cur)
            }
            "bias" -> {
                val cur = Bias(mx, action.slice(1..action.lastIndex).map { it.toFloat() }.toFloatArray())
                cur.compute()
                mx = cur.outputMatrix
                computations.add(cur)
            }
            else -> {
                val h = action[1].toInt()
                val k = action[2].toInt()
                val s = action[3].toInt()
                val p = action[4].toInt()
                val aMatrix = Matrix4D.read4DMx(
                    h,
                    mx.depth,
                    k,
                    action.slice(5..action.lastIndex).map { it.toFloat() }.toFloatArray()
                )
                val cur = Cnv(mx, action[0], h, k, s, p, aMatrix)
                cur.compute()
                mx = cur.outputMatrix
                computations.add(cur)
            }
        }
    }
    for (d in 0 until mx.depth) {
        for (i in 0 until mx.height) {
            for (j in 0 until mx.width) {
                print("${mx[d, i, j]} ")
            }
        }
    }
    println()
    val dOutput = Matrix3D.read3DMx(mx.height, mx.depth, readLine()!!.split(" ").map { it.toFloat() }.toFloatArray())
    computations[computations.lastIndex].dx = dOutput

    for (i in amountOfActions downTo 1) {
        computations[i].derivative(computations[i - 1])
    }
    for (d in 0 until computations[0].dx.depth) {
        for (ii in 0 until computations[0].dx.height) {
            for (j in 0 until computations[0].dx.width) {
                print("${computations[0].dx[d, ii, j]} ")
            }
        }
        println()
    }
    for (cp in computations) {
        when (cp) {
            is Bias -> {
                println(cp.dBs.joinToString(" "))
            }
            is Cnv -> {
                for (hh in 0 until cp.dWMatrix.h) {
                    for (d in 0 until cp.dWMatrix.depth) {
                        for (ii in 0 until cp.dWMatrix.height) {
                            for (j in 0 until cp.dWMatrix.width) {
                                print("${cp.dWMatrix[hh, d, ii, j]} ")
                            }
                        }
                    }
                }
                println()
            }
        }
    }

}