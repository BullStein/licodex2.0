package com.libras.translator

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.libras.translator.databinding.ActivityMainBinding
import okhttp3.*
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * App LIBRAS Translator - lado do celular.
 * --------------------------------------------
 * O celular só faz duas coisas: (1) captura a câmera e manda os frames
 * pro PC via WebSocket, (2) mostra o que o PC respondeu. TODO o
 * reconhecimento (MediaPipe + classificador) roda no PC, no
 * stream_server.py -- ver esse arquivo pro protocolo completo.
 *
 * Os botões BACKSPACE / SPACE / CLEAR são 100% locais: editam o texto
 * na tela sem precisar do servidor, porque a mão que segura o celular
 * geralmente não está livre pra fazer um "sinal de comando".
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private var webSocket: WebSocket? = null
    private val httpClient = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS) // WebSocket fica aberto indefinidamente
        .build()

    // Controla o ritmo de envio de frames pra não afogar a rede/servidor.
    private val sendingFrame = AtomicBoolean(false)
    private val lastSendTime = AtomicLong(0)
    private val minFrameIntervalMs = 80L // ~12 fps de envio (suficiente pra reconhecimento)
    private val jpegQuality = 70

    private var currentLetter: String = "-"
    private var history: StringBuilder = StringBuilder()

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                Toast.makeText(this, "Sem permissão de câmera, o app não funciona.", Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        binding.editIp.setText("192.168.0.42") // valor de exemplo, troque pelo IP real do seu PC
        binding.editPort.setText("8765")

        binding.btnConnect.setOnClickListener { toggleConnection() }
        binding.btnBackspace.setOnClickListener { onLocalCommand("BACKSPACE") }
        binding.btnSpace.setOnClickListener { onLocalCommand("SPACE") }
        binding.btnClear.setOnClickListener { onLocalCommand("CLEAR") }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }

        updateUi()
    }

    // ------------------------------------------------------------------
    // Conexão WebSocket
    // ------------------------------------------------------------------

    private fun toggleConnection() {
        if (webSocket != null) {
            disconnect()
        } else {
            connect()
        }
    }

    private fun connect() {
        val ip = binding.editIp.text.toString().trim()
        val port = binding.editPort.text.toString().trim()
        if (ip.isEmpty() || port.isEmpty()) {
            Toast.makeText(this, "Preencha IP e porta do servidor.", Toast.LENGTH_SHORT).show()
            return
        }
        val url = "ws://$ip:$port"
        val request = Request.Builder().url(url).build()

        setStatus("Conectando em $url...")
        webSocket = httpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                runOnUiThread {
                    setStatus("Conectado a $url")
                    binding.btnConnect.text = "Desconectar"
                }
            }

            override fun onMessage(ws: WebSocket, text: String) {
                handleServerMessage(text)
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                Log.e("LibrasWS", "Falha na conexão", t)
                runOnUiThread {
                    setStatus("Erro de conexão: ${t.message}")
                    binding.btnConnect.text = "Conectar"
                }
                webSocket = null
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                runOnUiThread {
                    setStatus("Desconectado")
                    binding.btnConnect.text = "Conectar"
                }
                webSocket = null
            }
        })
    }

    private fun disconnect() {
        webSocket?.close(1000, "Usuário desconectou")
        webSocket = null
        setStatus("Desconectado")
        binding.btnConnect.text = "Conectar"
    }

    private fun handleServerMessage(text: String) {
        try {
            val json = JSONObject(text)
            if (json.has("erro")) {
                Log.w("LibrasWS", "Erro do servidor: ${json.getString("erro")}")
                return
            }
            val letra = json.optString("letra", "-")
            val fonte = if (json.isNull("fonte")) null else json.optString("fonte", null)

            runOnUiThread {
                onLetterRecognized(letra, fonte)
            }
        } catch (e: Exception) {
            Log.e("LibrasWS", "Mensagem inválida do servidor: $text", e)
        }
    }

    // ------------------------------------------------------------------
    // Lógica de reconhecimento (comita no histórico só quando MUDA,
    // igual ao translator.py original)
    // ------------------------------------------------------------------

    private fun onLetterRecognized(letra: String, fonte: String?) {
        binding.textCurrentLetter.text = if (letra == "-" || letra == "?") "-" else letra
        binding.textFonte.text = when (fonte) {
            "movimento" -> "(movimento)"
            "estatica" -> "(estática)"
            else -> ""
        }

        if (letra != "-" && letra != "?" && letra != currentLetter) {
            history.append(letra)
            trimHistory()
            currentLetter = letra
            binding.textHistory.text = history.toString()
        } else if (letra == "-" || letra == "?") {
            currentLetter = "-"
        }
    }

    private fun onLocalCommand(command: String) {
        when (command) {
            "BACKSPACE" -> if (history.isNotEmpty()) history.deleteCharAt(history.length - 1)
            "SPACE" -> history.append(" ")
            "CLEAR" -> history.clear()
        }
        trimHistory()
        binding.textHistory.text = history.toString()
    }

    private fun trimHistory(maxLen: Int = 60) {
        if (history.length > maxLen) {
            val excess = history.length - maxLen
            history.delete(0, excess)
        }
    }

    private fun setStatus(text: String) {
        binding.textStatus.text = text
    }

    private fun updateUi() {
        binding.textCurrentLetter.text = "-"
        binding.textHistory.text = ""
        binding.textStatus.text = "Desconectado"
    }

    // ------------------------------------------------------------------
    // Câmera (CameraX) -> JPEG -> envia pro servidor
    // ------------------------------------------------------------------

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy -> analyzeFrame(imageProxy) }
                }

            // Câmera FRONTAL por padrão -- é a que o usuário vê enquanto faz o
            // sinal olhando pra própria tela. Troque pra DEFAULT_BACK_CAMERA
            // se preferir gravar com a traseira (melhor qualidade, mas sem
            // "espelho" pra se guiar).
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            } catch (e: Exception) {
                Log.e("LibrasCamera", "Falha ao iniciar câmera", e)
                Toast.makeText(this, "Não foi possível iniciar a câmera.", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()
        val ws = webSocket

        // Throttle: só manda frame se já passou o intervalo mínimo E não tem
        // outro frame ainda em voo (evita acumular backlog se a rede/PC
        // estiver mais lento que a câmera).
        if (ws == null || sendingFrame.get() || (now - lastSendTime.get()) < minFrameIntervalMs) {
            imageProxy.close()
            return
        }

        try {
            sendingFrame.set(true)
            lastSendTime.set(now)
            val jpegBytes = imageProxyToJpeg(imageProxy, jpegQuality)
            ws.send(jpegBytes.toByteString())
        } catch (e: Exception) {
            Log.e("LibrasCamera", "Falha ao converter/enviar frame", e)
        } finally {
            sendingFrame.set(false)
            imageProxy.close()
        }
    }

    /**
     * Converte o frame YUV_420_888 (formato nativo do CameraX) pra JPEG.
     * É uma conversão "boa o suficiente" pra reconhecimento de mão -- não
     * é pixel-perfect, mas mantém qualidade suficiente pro MediaPipe do
     * lado do servidor detectar a mão sem problema.
     */
    private fun imageProxyToJpeg(image: ImageProxy, quality: Int): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), quality, out)
        return out.toByteArray()
    }

    override fun onDestroy() {
        super.onDestroy()
        disconnect()
        cameraExecutor.shutdown()
    }
}
