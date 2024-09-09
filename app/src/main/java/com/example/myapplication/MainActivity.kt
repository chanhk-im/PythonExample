//package com.example.myapplication
//
//import android.graphics.BitmapFactory
//import android.os.Bundle
//import android.widget.ImageView
//import androidx.activity.ComponentActivity
//import androidx.compose.material3.Text
//import androidx.compose.runtime.Composable
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.tooling.preview.Preview
//import com.chaquo.python.Python
//import com.example.myapplication.ui.theme.MyApplicationTheme
//import android.util.Base64;
//
//
//class MainActivity : ComponentActivity() {
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        setContentView(R.layout.activity_main)
//
//        val imageView: ImageView = findViewById(R.id.imageView)
//
//        // 예제 이미지 데이터 (base64 인코딩된 문자열)
//        val imgBase64 = "iVBORw0KGgoAAAANSUhEUgAAAGUAAAA4CAIAAABrDqGxAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAKJSURBVHhe7ZfbcoMwDERD//+fU5VlNB4j2ZItE0+j89ASo8tqMYQc7/f7lZj5uf4nNtIvH+mXj/TLR/rlI/3ysalfx3FcR5ux7/7a07JN/dr2LTreL+O+oLDwHRRe8I7Pr8AhsYNQkMDiPIGlRKy/H1lHN54ivXcTildZ4mIXu84x+rN5FQz4pTFcasxrC6b7kRpzb5ICNTsDwZBKXKsRdPy6N4NxsSIWAdfo4DTtD6zP0N9faFlyX9mZ07Q/6HjetdYDAqXFADqlJTZOlYi6q8SGgKU0RujsrxVaSQ0bUYKzJeLiZ/G9f4UAd4jr8wkc3ATSpunp3I/a2bFTGhB3z9JEM95GGlUjKqtNMeKXNh7QskRYaEg1oq2N4b5MlaL17fhFfzmg7NHOapwt4fpc+Z5YabBQ6iSQKy42oHgxpjObtw2hdapA5TKSe1XpxoIiA/qB1nRcioZlPIwhCyomRICloEaZ22hawgLEyODvR8tsbd20DuiYpYeAmhrUC9BxI/ID7xNGTtM61rfB8Bpwh6EVdCQQIBJ8P1Lj2IIzsBi2AwdMQ6o2yEbjhVPOzE4Z513lV3XFRCYlDnMvaG8R4JdoTTcdWZYu9mGM3AvaW2iRjuc95d+5zukgBq59lhANT3w/WmxdDcyaV+Lwa4c94gWaB8zSgvd9/5qnfBRE7XHf8364qyV3pv4YsFJDFLOXX/T3GcvYKa2dJnij+/EZpwD1AtdnMz6/+LJ8LQ6/Bq4GsNyMzOaXZPn96Joftm5imSjD9wh37RQCLb0bs+uXt+AY4rCDfnVHAs8MtogAv76K9CuA//x7aAXpl4/0y0f65SP98pF++Ui/fKRfPtIvH+mXj/TLw+v1Cy0Q8iV6MPyRAAAAAElFTkSuQmCC"
//
//        // Python 인터프리터 초기화
//        val py = Python.getInstance()
//        val pyf = py.getModule("opencv_script")
//
//        // Python 함수 호출
//        val processedImgBase64 = pyf.callAttr("process_image", imgBase64).toString()
//        val decodedString = Base64.decode(processedImgBase64, Base64.DEFAULT)
//
//        // 이미지 뷰에 설정
//        imageView.setImageBitmap(BitmapFactory.decodeByteArray(decodedString, 0, decodedString.size))
//    }
//}

package com.example.myapplication

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import android.view.MotionEvent
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.chaquo.python.Python
import java.io.ByteArrayOutputStream
import java.io.FileOutputStream
import java.text.SimpleDateFormat


class MainActivity : AppCompatActivity() {

    // storage 권한 처리에 필요한 변수
    val CAMERA = arrayOf(Manifest.permission.CAMERA)
    val STORAGE = arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE,
        Manifest.permission.WRITE_EXTERNAL_STORAGE)
    val CAMERA_CODE = 98
    val STORAGE_CODE = 99
    val py = Python.getInstance()
    val pyf = py.getModule("opencv_script")
    val pyf2 = py.getModule("Brick_Line_Detection")
    val a = pyf2.callAttr("main", "Line_Data.jpg")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 카메라
        val camera = findViewById<Button>(R.id.camera)
        camera.setOnClickListener {
            CallCamera()
        }

        // 사진 저장
        val picture = findViewById<Button>(R.id.picture)
        picture.setOnClickListener {
            GetAlbum()
        }


    }

    // 카메라 권한, 저장소 권한
    // 요청 권한
    override fun onRequestPermissionsResult(requestCode: Int,
                                            permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        when(requestCode){
            CAMERA_CODE -> {
                for (grant in grantResults){
                    if(grant != PackageManager.PERMISSION_GRANTED){
                        Toast.makeText(this, "카메라 권한을 승인해 주세요", Toast.LENGTH_LONG).show()
                    }
                }
            }
            STORAGE_CODE -> {
                for(grant in grantResults){
                    if(grant != PackageManager.PERMISSION_GRANTED){
                        Toast.makeText(this, "저장소 권한을 승인해 주세요", Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
    }

    // 다른 권한등도 확인이 가능하도록
    fun checkPermission(permissions: Array<out String>, type:Int):Boolean{
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
            for (permission in permissions){
                if (ContextCompat.checkSelfPermission(this, permission)
                    != PackageManager.PERMISSION_GRANTED){
                    Toast.makeText(this, permissions.first(), Toast.LENGTH_LONG).show()
                    ActivityCompat.requestPermissions(this, permissions, type)
//                    return false;
                }
            }
        }
        return true
    }

    // 카메라 촬영 - 권한 처리
    fun CallCamera(){
        if(checkPermission(CAMERA, CAMERA_CODE) && checkPermission(STORAGE, STORAGE_CODE)){
            val itt = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(itt, CAMERA_CODE)
        }
    }

    // 사진 저장
    fun saveFile(fileName:String, mimeType:String, bitmap: Bitmap):Uri?{

        var CV = ContentValues()

        // MediaStore 에 파일명, mimeType 을 지정
        CV.put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
        CV.put(MediaStore.Images.Media.MIME_TYPE, mimeType)

        // 안정성 검사
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q){
            CV.put(MediaStore.Images.Media.IS_PENDING, 1)
        }

        // MediaStore 에 파일을 저장
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, CV)
        if(uri != null){
            var scriptor = contentResolver.openFileDescriptor(uri, "w")

            val fos = FileOutputStream(scriptor?.fileDescriptor)

            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)
            fos.close()

            if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q){
                CV.clear()
                // IS_PENDING 을 초기화
                CV.put(MediaStore.Images.Media.IS_PENDING, 0)
                contentResolver.update(uri, CV, null, null)
            }
        }
        return uri
    }

    // 결과
    @SuppressLint("ClickableViewAccessibility")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        val imageView = findViewById<ImageView>(R.id.avatars)

        if(resultCode == Activity.RESULT_OK){
            when(requestCode){
                CAMERA_CODE -> {
                    if(data?.extras?.get("data") != null){
                        val img = data?.extras?.get("data") as Bitmap
                        val uri = saveFile(RandomFileName(), "image/jpeg", img)
                        println(uri.toString())
                        val imgBase64 = BitmapToString(img)
                        val processedImgBase64 = pyf.callAttr("process_image", imgBase64).toString()
                        val decodedString = Base64.decode(processedImgBase64, Base64.DEFAULT)

                        // 이미지 뷰에 설정
                        imageView.setImageBitmap(BitmapFactory.decodeByteArray(decodedString, 0, decodedString.size))
                        imageView.setOnTouchListener { _, event ->
                            when (event.action) {
                                MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                                    // 터치 좌표를 가져옴
                                    val x = event.x
                                    val y = event.y

                                    // 좌표를 로그로 출력
                                    println("Touched at: ($x, $y)")
                                }
                            }
                            true
                        }
                    }
                }
                STORAGE_CODE -> {
                    data?.data?.let { uri ->
                        val bitmap = uriToBitmap(uri) as Bitmap
                        val imgBase64 = BitmapToString(bitmap)
                        val processedImgBase64 = pyf.callAttr("process_image", imgBase64).toString()
                        val decodedString = Base64.decode(processedImgBase64, Base64.DEFAULT)

                        // 이미지 뷰에 설정
                        imageView.setImageBitmap(BitmapFactory.decodeByteArray(decodedString, 0, decodedString.size))
                    }
                    val uri = data?.data
                }
            }
        }
    }

    fun BitmapToString(bitmap: Bitmap): String {
        val baos = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 70, baos)
        val bytes = baos.toByteArray()
        val temp = Base64.encodeToString(bytes, Base64.DEFAULT)
        return temp
    }

    private fun uriToBitmap(uri: Uri): Bitmap? {
        return try {
            val parcelFileDescriptor = contentResolver.openFileDescriptor(uri, "r")
            val fileDescriptor = parcelFileDescriptor?.fileDescriptor
            val image = BitmapFactory.decodeFileDescriptor(fileDescriptor)
            parcelFileDescriptor?.close()
            image
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // 파일명을 날짜 저장
    fun RandomFileName() : String{
        val fileName = SimpleDateFormat("yyyyMMddHHmmss").format(System.currentTimeMillis())
        return fileName
    }

    // 갤러리 취득
    fun GetAlbum(){
        if(checkPermission(STORAGE, STORAGE_CODE)){
            val itt = Intent(Intent.ACTION_PICK)
            itt.type = MediaStore.Images.Media.CONTENT_TYPE
            startActivityForResult(itt, STORAGE_CODE)
        }
    }
}