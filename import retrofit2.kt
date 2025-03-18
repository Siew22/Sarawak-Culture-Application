import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET

// 定义数据模型
data class Place(
    val id: Int,
    val name: String,
    val description: String
)

// 定义 Retrofit API 接口
interface ApiService {
    @GET("places") // 这个路径对应 FastAPI 的 /places
    suspend fun getPlaces(): List<Place>
}

// 创建 Retrofit 实例
object RetrofitClient {
    private const val BASE_URL = "http://0.0.0.0:8800"

    val api: ApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ApiService::class.java)
    }
}
