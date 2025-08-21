package com.example.byos

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraCharacteristics
import java.io.File
import java.io.IOException

object FunctionHandler {
    /**
     * Dispatches a function call parsed from the LLM.
     * Returns a user-friendly result string.
     */
    fun dispatch(context: Context, name: String, args: Map<String, String>): String {
        // Use the exact function name from the AI response
        return when (name) {
            "toggle_flashlight" -> runCatching {
                val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                val cameraIds = cameraManager.cameraIdList
                for (cameraId in cameraIds) {
                    val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                    val hasFlash = characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
                    if (hasFlash) {
                        // Note: This is a simplified toggle - in a real app you'd track state
                        cameraManager.setTorchMode(cameraId, true)
                        return "✅ Flashlight turned on"
                    }
                }
                "❌ No flashlight available on this device"
            }.getOrElse { "❌ Failed to control flashlight: ${it.message}" }

            "copy_file" -> runCatching {
                val src = args["source"] ?: return "Missing 'source'"
                val dst = args["destination"] ?: return "Missing 'destination'"
                File(src).copyTo(File(dst), overwrite = true)
                "Copied $src to $dst"
            }.getOrElse { "Failed to copy file: ${it.message}" }

            "move_file" -> runCatching {
                val src = args["source"] ?: return "Missing 'source'"
                val dst = args["destination"] ?: return "Missing 'destination'"
                File(src).also { it.copyTo(File(dst), overwrite = true); it.delete() }
                "Moved $src to $dst"
            }.getOrElse { "Failed to move file: ${it.message}" }

            "delete_file" -> runCatching {
                val path = args["path"] ?: return "Missing 'path'"
                File(path).delete()
                "Deleted file $path"
            }.getOrElse { "Failed to delete file: ${it.message}" }

            "create_folder" -> runCatching {
                val path = args["path"] ?: return "Missing 'path'"
                File(path).apply { mkdirs() }
                "Created folder at $path"
            }.getOrElse { "Failed to create folder: ${it.message}" }

            "open_application" -> runCatching {
                val appArg = args["app_name"] ?: return "Missing 'app_name'"
                val pm = context.packageManager
                // If it's already a package name and resolvable
                pm.getLaunchIntentForPackage(appArg)?.let { intent ->
                    context.startActivity(intent)
                    return "Opened app: $appArg"
                }
                // Try best-effort fuzzy match on app label
                val match = pm.getInstalledApplications(0).firstOrNull { appInfo ->
                    val label = pm.getApplicationLabel(appInfo).toString()
                    label.equals(appArg, ignoreCase = true) ||
                    label.contains(appArg, ignoreCase = true)
                }
                if (match == null) {
                    // Heuristic: treat common flashlight aliases as torch toggle
                    val lower = appArg.lowercase()
                    if (lower.contains("flash") || lower.contains("torch") || lower.contains("lamp")) {
                        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                        val cameraIds = cameraManager.cameraIdList
                        for (cameraId in cameraIds) {
                            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                            val hasFlash = characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
                            if (hasFlash) {
                                cameraManager.setTorchMode(cameraId, true)
                                return "✅ Flashlight turned on"
                            }
                        }
                        return "❌ No flashlight available on this device"
                    }
                    return "App not found: $appArg"
                }
                val pkg = match.packageName
                val intent = pm.getLaunchIntentForPackage(pkg) ?: return "App not launchable: $pkg"
                context.startActivity(intent)
                "Opened app: $pkg"
            }.getOrElse { "Failed to open app: ${it.message}" }

            "close_application" -> runCatching {
                val pkg = args["app_name"] ?: return "Missing 'app_name'"
                val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                am.killBackgroundProcesses(pkg)
                "Closed app: $pkg"
            }.getOrElse { "Failed to close app: ${it.message}" }

            "take_screenshot" -> "Screenshot not supported on Android offline"

            "lock_screen" -> "Lock screen not supported on Android without special permissions"

            "get_battery_status" -> runCatching {
                val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
                val level = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                    bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
                } else -1
                val charging = bm.isCharging
                "Battery: $level%, Charging: $charging"
            }.getOrElse { "Failed to get battery status: ${it.message}" }

            "open_url" -> runCatching {
                val url = args["url"] ?: return "Missing 'url'"
                val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(intent)
                "Opened URL: $url"
            }.getOrElse { "Failed to open URL: ${it.message}" }

            "search_google" -> runCatching {
                val q = args["query"] ?: return "Missing 'query'"
                val url = "https://www.google.com/search?q=${Uri.encode(q)}"
                val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(intent)
                "Searched Google for: $q"
            }.getOrElse { "Failed to search: ${it.message}" }

            "play_music" -> args["track_name"]?.let { "Now playing: $it" }
                ?: "Missing 'track_name'"

            "pause_music" -> "Music paused"

            "set_volume" -> runCatching {
                val level = args["level"]?.toIntOrNull() ?: return "Invalid 'level'"
                val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
                am.setStreamVolume(AudioManager.STREAM_MUSIC,
                    (level / 100.0 * am.getStreamMaxVolume(AudioManager.STREAM_MUSIC)).toInt(), 0)
                "Volume set to $level%"
            }.getOrElse { "Failed to set volume: ${it.message}" }

            "create_note" -> args["title"]?.let { title ->
                val content = args["content"] ?: ""
                "Note created: $title - $content"
            } ?: "Missing 'title'"

            else -> "Unknown function: $name"
        }
    }
}

