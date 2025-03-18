// 1. 添加依赖
implementation 'com.squareup.retrofit2:retrofit:2.9.0'
implementation 'com.squareup.retrofit2:converter-gson:2.9.0'

// 2. 定义 API 接口
interface TravelApi {
    @GET("itineraries/{user_id}")
    suspend fun getItineraries(
        @Path("user_id") userId: Int,
        @Header("Authorization") token: String
    ): Response<ItineraryResponse>

    @POST("token")
    @FormUrlEncoded
    suspend fun login(
        @Field("username") username: String,
        @Field("password") password: String
    ): Response<TokenResponse>
}

data class ItineraryResponse(val itineraries: List<ItineraryDay>)
data class ItineraryDay(val day: Int, val schedule: Map<String, Map<String, List<String>>>)
data class TokenResponse(val access_token: String, val token_type: String)

// 3. 配置 Retrofit
val retrofit = Retrofit.Builder()
    .baseUrl("http://your-api-url:8800/")
    .addConverterFactory(GsonConverterFactory.create())
    .build()

val api = retrofit.create(TravelApi::class.java)

// 4. 调用 API
CoroutineScope(Dispatchers.IO).launch {
    try {
        val tokenResponse = api.login("username", "password")
        val token = "Bearer ${tokenResponse.body()?.access_token}"
        val response = api.getItineraries(1, token)
        if (response.isSuccessful) {
            val itineraries = response.body()?.itineraries
            withContext(Dispatchers.Main) {
                // 更新UI
            }
        }
    } catch (e: Exception) {
        // 处理错误
    }
}