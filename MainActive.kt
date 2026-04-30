package com.example.smsagent

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.provider.Telephony
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import com.example.smsagent.ui.theme.SmsagentTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import android.content.Intent
import androidx.compose.runtime.collectAsState
import android.provider.Settings

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SmsagentTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    SmsScanScreen(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(innerPadding)
                            .padding(16.dp)
                    )
                }
            }
        }
    }
}

private data class SmsRow(
    val id: Long,
    val address: String,
    val body: String,
    val date: Long,
)

private fun loadRecentSms(context: android.content.Context, limit: Int = 50): List<SmsRow> {
    val resolver = context.contentResolver
    val uri = Telephony.Sms.Inbox.CONTENT_URI
    val proj = arrayOf(
        Telephony.Sms._ID,
        Telephony.Sms.ADDRESS,
        Telephony.Sms.BODY,
        Telephony.Sms.DATE,
    )
    val list = mutableListOf<SmsRow>()
    resolver.query(uri, proj, null, null, "${Telephony.Sms.DATE} DESC")?.use { c ->
        val idIdx = c.getColumnIndexOrThrow(Telephony.Sms._ID)
        val addrIdx = c.getColumnIndexOrThrow(Telephony.Sms.ADDRESS)
        val bodyIdx = c.getColumnIndexOrThrow(Telephony.Sms.BODY)
        val dateIdx = c.getColumnIndexOrThrow(Telephony.Sms.DATE)
        var n = 0
        while (c.moveToNext() && n < limit) {
            list.add(
                SmsRow(
                    id = c.getLong(idIdx),
                    address = c.getString(addrIdx).orEmpty(),
                    body = c.getString(bodyIdx).orEmpty(),
                    date = c.getLong(dateIdx),
                )
            )
            n++
        }
    }
    return list
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ScanResultPanel(raw: String) {
    val trimmed = raw.trim()
    if (trimmed.isEmpty() || trimmed == "（尚未送出）") {
        Text(trimmed.ifEmpty { "（尚未送出）" })
        return
    }

    if (!trimmed.trimStart().startsWith("{")) {
        Text(trimmed)
        return
    }

    runCatching {
        val root = JSONObject(trimmed)
        val report = root.optJSONObject("report") ?: JSONObject(trimmed)

        val verdict = report.optString("verdict", "unknown")
        val risk = report.optInt("risk_score_total", -1)
        val inj = report.optInt("injection_score", -1)
        val scam = report.optInt("scam_score", -1)

        val verdictColor = when (verdict.lowercase()) {
            "block" -> MaterialTheme.colorScheme.error
            "quarantine" -> MaterialTheme.colorScheme.tertiary
            else -> MaterialTheme.colorScheme.primary
        }

        val labels = report.optJSONArray("labels")
        val signals = report.optJSONArray("signals")
        val archive = report.optString("archive_note", "")
        val memory = report.optString("sanitized_memory", "")

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("審查結果", style = MaterialTheme.typography.titleMedium)

            Text(
                text = verdict.uppercase(),
                style = MaterialTheme.typography.headlineSmall,
                color = verdictColor
            )

            Text("總風險：$risk　注入：$inj　詐騙：$scam")
            val fusion = report.optJSONObject("risk_fusion")
            if (fusion != null) {
                val rLlm = fusion.optInt("r_llm", -1)
                val sTox = fusion.optInt("s_tox", -1)
                val riskFus = fusion.optInt("risk_total", -1)
                Text("R_LLM：$rLlm　S_tox：$sTox　融合：$riskFus")
            }
            val fusedOnly = report.optInt("risk_score_total_fused", -1)
            if (fusedOnly >= 0) {
                Text("risk_score_total_fused：$fusedOnly")
            }

            if (labels != null && labels.length() > 0) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    for (i in 0 until labels.length()) {
                        val t = labels.optString(i)
                        if (t.isNotBlank()) {
                            AssistChip(onClick = {}, label = { Text(t) })
                        }
                    }
                }
            }

            if (signals != null && signals.length() > 0) {
                Text("可疑徵兆", style = MaterialTheme.typography.titleSmall)
                signals.forEachSignal { type, sev, ev ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                    ) {
                        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text("$type（$sev）", style = MaterialTheme.typography.labelLarge)
                            Text(ev, style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
            }

            HorizontalDivider()

            if (archive.isNotBlank()) {
                Text("摘要", style = MaterialTheme.typography.titleSmall)
                Text(archive, style = MaterialTheme.typography.bodyMedium)
            }

            if (memory.isNotBlank()) {
                Text("去毒可檢索摘要", style = MaterialTheme.typography.titleSmall)
                Text(
                    memory,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.heightIn(max = 220.dp)
                )
            }
        }
    }.getOrElse { e ->
        Text("JSON 解析失敗：${e.message}\n\n$trimmed")
    }
}

private data class ChatBubble(val isUser: Boolean, val text: String)

private inline fun JSONArray.forEachSignal(block: (type: String, severity: String, evidence: String) -> Unit) {
    for (i in 0 until length()) {
        val o = optJSONObject(i) ?: continue
        block(
            o.optString("type"),
            o.optString("severity"),
            o.optString("evidence")
        )
    }
}

private const val API_BASE = "http://127.0.0.1:8787"

private val jsonMedia = "application/json; charset=utf-8".toMediaType()

private val http = OkHttpClient.Builder()
    .connectTimeout(30, TimeUnit.SECONDS)
    .readTimeout(300, TimeUnit.SECONDS)
    .writeTimeout(30, TimeUnit.SECONDS)
    .build()

@Composable
fun SmsScanScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val notifs by NotificationInbox.items.collectAsState()
    var text by remember { mutableStateOf("") }
    /** 最近一次從通知帶入時的 packageName，送審 / Chat 會一併給後端算 risk_user */
    var contextSource by remember { mutableStateOf<String?>(null) }
    var result by remember { mutableStateOf("（尚未送出）") }
    var showSmsPicker by remember { mutableStateOf(false) }
    var smsRows by remember { mutableStateOf<List<SmsRow>>(emptyList()) }
    var showInbox by remember { mutableStateOf(false) }
    var selectedNotif by remember { mutableStateOf<NotifItem?>(null) }
    var showNotifDetail by remember { mutableStateOf(false) }
    var showChat by remember { mutableStateOf(false) }
    var chatInput by remember { mutableStateOf("") }
    val chatMessages = remember { mutableStateListOf<ChatBubble>() }
    val chatListState = rememberLazyListState()
    var chatBusy by remember { mutableStateOf(false) }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            smsRows = loadRecentSms(context)
            showSmsPicker = true
        } else {
            result = "未授予讀取簡訊權限，無法自動擷取。"
        }
    }

    fun openSmsPicker() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_SMS) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            smsRows = loadRecentSms(context)
            showSmsPicker = true
        } else {
            permissionLauncher.launch(Manifest.permission.READ_SMS)
        }
    }

    Column(
        modifier = modifier.verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("貼上簡訊全文，按「送審」會呼叫電腦 API")

        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            modifier = Modifier.fillMaxWidth(),
            minLines = 6,
            label = { Text("簡訊內容") }
        )

        // 獨立的按鈕：負責擷取簡訊
        Button(onClick = {
            val i = Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
            context.startActivity(i)
        }) {
            Text("開啟通知存取權")
        }
        Button(onClick = { showInbox = true }) {
            Text("打開收件匣（通知）")
        }
        if (showInbox) {
            AlertDialog(
                onDismissRequest = { showInbox = false },
                title = { Text("通知收件匣（點一則查看）") },
                text = {
                    LazyColumn(modifier = Modifier.heightIn(max = 450.dp)) {
                        items(notifs, key = { it.key }) { item ->
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable {
                                        selectedNotif = item
                                        showNotifDetail = true
                                    }
                                    .padding(vertical = 10.dp)
                            ) {
                                Text(item.packageName, style = MaterialTheme.typography.labelSmall)
                                if (item.title.isNotBlank()) {
                                    Text(item.title, style = MaterialTheme.typography.labelLarge)
                                }
                                Text(
                                    item.content.lineSequence().firstOrNull().orEmpty().take(60) +
                                            if (item.content.length > 60) "…" else "",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                            HorizontalDivider()
                        }
                    }
                },
                confirmButton = {
                    TextButton(onClick = { NotificationInbox.clear() }) { Text("清空") }
                },
                dismissButton = {
                    TextButton(onClick = { showInbox = false }) { Text("關閉") }
                }
            )
        }
        if (showNotifDetail && selectedNotif != null) {
            val item = selectedNotif!!
            AlertDialog(
                onDismissRequest = { showNotifDetail = false },
                title = { Text("通知全文") },
                text = { Text(item.content) },
                confirmButton = {
                    TextButton(onClick = {
                        text = item.content
                        contextSource = item.packageName
                        result = "已帶入通知內容，可按「送審」。"
                        showNotifDetail = false
                        showInbox = false
                    }) { Text("帶入") }
                },
                dismissButton = {
                    TextButton(onClick = { showNotifDetail = false }) { Text("返回") }
                }
            )
        }

        Button(onClick = { NotificationInbox.clear() }) {
            Text("清空通知收件匣")
        }
        Text("最近通知（點一下帶入送審）")

        LazyColumn(modifier = Modifier.heightIn(max = 260.dp)) {
            items(notifs, key = { it.key }) { item ->
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            text = item.content
                            contextSource = item.packageName
                            result = "已帶入通知內容（${item.packageName}），可按「送審」。"
                        }
                        .padding(vertical = 8.dp)
                ) {
                    Text(item.packageName, style = MaterialTheme.typography.labelSmall)
                    if (item.title.isNotBlank()) {
                        Text(item.title, style = MaterialTheme.typography.labelLarge)
                    }
                    Text(
                        item.content.take(80) + if (item.content.length > 80) "…" else "",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                HorizontalDivider()
            }
        }
        // 獨立的按鈕：負責發送請求給 API
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            Button(
                onClick = {
                    scope.launch {
                        result = "送出中…（若很久，通常是 Ollama 還在推理）"
                        result = runCatching { scanSms(text, contextSource) }
                            .fold(
                                onSuccess = { it },
                                onFailure = { e -> "錯誤：${e.javaClass.simpleName}: ${e.message}" }
                            )
                    }
                },
                enabled = text.isNotBlank(),
                modifier = Modifier.weight(1f)
            ) {
                Text("送審")
            }
            Button(
                onClick = {
                    if (chatMessages.isEmpty()) {
                        chatMessages.add(
                            ChatBubble(
                                false,
                                "你好，我是反詐 Agent。請在下方輸入想問的問題（我會依上方「待審文字」與後端審查結果回答）。"
                            )
                        )
                    }
                    showChat = true
                },
                enabled = text.isNotBlank(),
                modifier = Modifier.weight(1f)
            ) {
                Text("問 Agent")
            }
        }
        if (contextSource != null) {
            Text("來源：$contextSource", style = MaterialTheme.typography.labelSmall)
        }

        ScanResultPanel(result)
    }

    if (showChat) {
        Dialog(onDismissRequest = { showChat = false }) {
            Surface(shape = MaterialTheme.shapes.extraLarge) {
                Column(
                    modifier = Modifier
                        .widthIn(max = 520.dp)
                        .height(480.dp)
                        .padding(16.dp)
                ) {
                    Text("與 Agent 對話", style = MaterialTheme.typography.titleMedium)
                    Text(
                        "建議先「送審」再問：後端會快取審查 JSON，追問時只跑對答模型較快。",
                        style = MaterialTheme.typography.bodySmall
                    )
                    if (chatBusy) {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        Text("Agent 生成中（依模型與硬體，可能需數十秒～數分鐘）…", style = MaterialTheme.typography.labelSmall)
                    }
                    LazyColumn(
                        state = chatListState,
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth()
                            .padding(vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        items(chatMessages.size) { i ->
                            val b = chatMessages[i]
                            val bg = if (b.isUser) MaterialTheme.colorScheme.primaryContainer
                            else MaterialTheme.colorScheme.surfaceVariant
                            Surface(color = bg, shape = MaterialTheme.shapes.medium) {
                                Text(
                                    b.text,
                                    modifier = Modifier.padding(12.dp),
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                        }
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        OutlinedTextField(
                            value = chatInput,
                            onValueChange = { chatInput = it },
                            modifier = Modifier.weight(1f),
                            maxLines = 3,
                            label = { Text("輸入問題") }
                        )
                        Button(
                            onClick = {
                                val q = chatInput.trim()
                                if (q.isEmpty() || text.isBlank() || chatBusy) return@Button
                                chatMessages.add(ChatBubble(true, q))
                                chatInput = ""
                                chatBusy = true
                                scope.launch {
                                    try {
                                        val reply = runCatching { chatAsk(text, q, contextSource) }.fold(
                                            onSuccess = { it },
                                            onFailure = { e -> "錯誤：${e.javaClass.simpleName}: ${e.message}" }
                                        )
                                        chatMessages.add(ChatBubble(false, reply))
                                    } finally {
                                        chatBusy = false
                                    }
                                }
                            },
                            enabled = chatInput.isNotBlank() && text.isNotBlank() && !chatBusy
                        ) { Text("送出") }
                    }
                    TextButton(onClick = { showChat = false }, modifier = Modifier.align(Alignment.End)) {
                        Text("關閉")
                    }
                }
            }
        }
    }

    if (showSmsPicker) {
        AlertDialog(
            onDismissRequest = { showSmsPicker = false },
            title = { Text("選擇一則簡訊") },
            text = {
                LazyColumn(modifier = Modifier.heightIn(max = 400.dp)) {
                    items(smsRows, key = { it.id }) { row ->
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    text = row.body
                                    contextSource = row.address.takeIf { it.isNotBlank() }
                                    showSmsPicker = false
                                    result = "已擷取簡訊內容，可按「送審」。"
                                }
                                .padding(vertical = 8.dp)
                        ) {
                            Text(row.address, style = MaterialTheme.typography.labelLarge)
                            Text(
                                row.body.take(80) + if (row.body.length > 80) "…" else "",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showSmsPicker = false }) { Text("關閉") }
            }
        )
    }
}

private suspend fun scanSms(sms: String, source: String?): String = withContext(Dispatchers.IO) {
    // ⚠️ 注意：如果是用 Android 模擬器測試，請將 API_BASE 改為 http://10.0.2.2:8787
    // ⚠️ 如果是實體手機，請填寫你電腦在區域網路的 IP (例如 192.168.X.X)
    val url = "$API_BASE/v1/sms/scan"
    val body = JSONObject()
    body.put("text", sms)
    if (!source.isNullOrBlank()) body.put("source", source)
    val bodyJson = body.toString().toRequestBody(jsonMedia)

    val req = Request.Builder()
        .url(url)
        .post(bodyJson)
        .build()

    http.newCall(req).execute().use { resp ->
        val respBody = resp.body?.string().orEmpty()
        if (!resp.isSuccessful) {
            return@withContext "HTTP ${resp.code}：$respBody"
        }
        return@withContext respBody
    }
}

private suspend fun chatAsk(corpus: String, question: String, source: String?): String = withContext(Dispatchers.IO) {
    val url = "$API_BASE/v1/chat/ask"
    val body = JSONObject()
    body.put("text", corpus)
    body.put("question", question)
    body.put("ingest", false)
    if (!source.isNullOrBlank()) body.put("source", source)
    val bodyJson = body.toString().toRequestBody(jsonMedia)
    val req = Request.Builder().url(url).post(bodyJson).build()
    http.newCall(req).execute().use { resp ->
        val respBody = resp.body?.string().orEmpty()
        if (!resp.isSuccessful) {
            return@withContext "HTTP ${resp.code}：$respBody"
        }
        return@withContext runCatching {
            val root = JSONObject(respBody)
            val rawAns = root.opt("answer")
            val s = when {
                rawAns == null || rawAns === JSONObject.NULL -> ""
                rawAns is String -> rawAns.trim()
                else -> rawAns.toString().trim()
            }
            if (s.isNotEmpty() && !s.equals("null", ignoreCase = true)) {
                s
            } else {
                val rep = root.optJSONObject("report")
                buildString {
                    append("（後端 answer 為空或 null。請確認已重啟 uvicorn 並使用最新 server.py / agent.py。）\n")
                    if (rep != null) {
                        append("verdict=").append(rep.optString("verdict", "?"))
                        append(" risk=").append(rep.optInt("risk_score_total", -1))
                    }
                }
            }
        }.getOrDefault(respBody)
    }
}