package com.example.byos

import com.google.gson.Gson

object FunctionRegistry {
    data class ParamSchema(
        val type: String,
        val properties: Map<String, Map<String, String>> = emptyMap(),
        val required: List<String> = emptyList()
    )

    data class Tool(
        val name: String,
        val description: String,
        val parameters: ParamSchema
    )

    private val tools: List<Tool> = listOf(
        Tool(
            name = "toggle_flashlight",
            description = "Turns on the device flashlight if available.",
            parameters = ParamSchema(type = "object")
        ),
        Tool(
            name = "open_application",
            description = "Opens an application by name or package.",
            parameters = ParamSchema(
                type = "object",
                properties = mapOf(
                    "app_name" to mapOf("type" to "string")
                ),
                required = listOf("app_name")
            )
        ),
        Tool(
            name = "close_application",
            description = "Closes an application by package name.",
            parameters = ParamSchema(
                type = "object",
                properties = mapOf(
                    "app_name" to mapOf("type" to "string")
                ),
                required = listOf("app_name")
            )
        ),
        Tool(
            name = "copy_file",
            description = "Copies a file from source to destination.",
            parameters = ParamSchema(
                type = "object",
                properties = mapOf(
                    "source" to mapOf("type" to "string"),
                    "destination" to mapOf("type" to "string")
                ),
                required = listOf("source", "destination")
            )
        ),
        Tool(
            name = "move_file",
            description = "Moves a file from source to destination.",
            parameters = ParamSchema(
                type = "object",
                properties = mapOf(
                    "source" to mapOf("type" to "string"),
                    "destination" to mapOf("type" to "string")
                ),
                required = listOf("source", "destination")
            )
        ),
        Tool(
            name = "delete_file",
            description = "Deletes a file by path.",
            parameters = ParamSchema(
                type = "object",
                properties = mapOf(
                    "path" to mapOf("type" to "string")
                ),
                required = listOf("path")
            )
        ),
        Tool(
            name = "create_folder",
            description = "Creates a folder (and parents) at the specified path.",
            parameters = ParamSchema(
                type = "object",
                properties = mapOf(
                    "path" to mapOf("type" to "string")
                ),
                required = listOf("path")
            )
        ),
        Tool(
            name = "get_battery_status",
            description = "Gets current battery percentage and charging state.",
            parameters = ParamSchema(type = "object")
        ),
        Tool(
            name = "open_url",
            description = "Opens a URL in the browser.",
            parameters = ParamSchema(
                type = "object",
                properties = mapOf(
                    "url" to mapOf("type" to "string")
                ),
                required = listOf("url")
            )
        ),
        Tool(
            name = "search_google",
            description = "Performs a Google search in the browser.",
            parameters = ParamSchema(
                type = "object",
                properties = mapOf(
                    "query" to mapOf("type" to "string")
                ),
                required = listOf("query")
            )
        ),
        Tool(
            name = "play_music",
            description = "Simulates playing a music track.",
            parameters = ParamSchema(
                type = "object",
                properties = mapOf(
                    "track_name" to mapOf("type" to "string")
                ),
                required = listOf("track_name")
            )
        ),
        Tool(
            name = "pause_music",
            description = "Simulates pausing music playback.",
            parameters = ParamSchema(type = "object")
        ),
        Tool(
            name = "set_volume",
            description = "Sets media volume as a percentage (0-100).",
            parameters = ParamSchema(
                type = "object",
                properties = mapOf(
                    "level" to mapOf("type" to "integer")
                ),
                required = listOf("level")
            )
        ),
        Tool(
            name = "create_note",
            description = "Creates a simple note with title and content.",
            parameters = ParamSchema(
                type = "object",
                properties = mapOf(
                    "title" to mapOf("type" to "string"),
                    "content" to mapOf("type" to "string")
                ),
                required = listOf("title")
            )
        )
    )

    @JvmStatic
    fun toolsJson(): String = Gson().toJson(tools)
}


