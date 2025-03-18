import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            TravelApp()
        }
    }
}

@Composable
fun TravelApp() {
    var places by remember { mutableStateOf<List<Place>>(emptyList()) }

    // 获取 API 数据
    LaunchedEffect(Unit) {
        try {
            places = RetrofitClient.api.getPlaces()
        } catch (e: Exception) {
            Log.e("API_ERROR", "获取数据失败: ${e.message}")
        }
    }

    Scaffold(topBar = { TopAppBar(title = { Text("AI 旅行助手") }) }) { padding ->
        LazyColumn(modifier = Modifier.padding(padding)) {
            items(places) { place ->
                Card(modifier = Modifier.padding(8.dp)) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(place.name, style = MaterialTheme.typography.headlineMedium)
                        Text(place.description, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
    }
}
