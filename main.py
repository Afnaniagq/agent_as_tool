from dotenv import load_dotenv
from agents import Agent ,AsyncOpenAI ,OpenAIChatCompletionsModel ,RunConfig ,Runner ,enable_verbose_stdout_logging
import os
import asyncio
from openai.types.responses import ResponseTextDeltaEvent


load_dotenv()

#for debugging or tracing:
#enable_verbose_stdout_logging()

async def main():
    
    MODEL_NAME ="gemini-2.0-flash"
    GEMINI_API_KEY =os.getenv("GEMINI_API_KEY")



    external_client =AsyncOpenAI(
        api_key =GEMINI_API_KEY,
        base_url ="https://generativelanguage.googleapis.com/v1beta/openai/",

    )

    model =OpenAIChatCompletionsModel(
        model =MODEL_NAME,
        openai_client =external_client,
    )

    # Specialist agents (work as tools)
    turkish_agent = Agent(
        name="Turkish Agent",
        instructions="translate the user query to turkish",
        model =model,
    )

    chinese_agent = Agent(
        name="Chinese Agent",
        instructions="translate the user query to chinese",
        model =model,
    )

    arabic_agent = Agent(
        name="Arabic Agent",
        instructions="translate the user query to arabic",
        model =model,
    )
    
    config =RunConfig(
        model =model,
        model_provider =external_client,
        tracing_disabled =True,
    )

    # main (Manager agent) with specialist agents as tools
    translator_agent =Agent(
        name ="Translator_agent",
        instructions = (
            "You are a translation agent specialized in Turkish, Chinese, and Arabic.\n"
            "When the user asks for a translation, identify the correct target language and translate ONLY into that language.\n"
            "\n"
            "Your response must be exactly one friendly, beautifully phrased sentence in this format:\n"
            "\"<Original word or phrase> in <Target Language> is \\\"<Translated Word or phrase>\\\" <Emoji>\"\n"
            "\n"
            "💡 Examples:\n"
            "- \"Hello good morning \" in Turkish is \"Merhaba Günaydin\" 😊\n"
            "- \"Goodbye\" in Arabic is \"مع السلامة\" 😢\n"
            "\n"
            "✅ Use quotation marks around both the original and translated words.\n"
            "✅ Pick an emoji that reflects the emotional tone of the word (happy or sad).\n"
            "If user did not give the translate language ask them to specify it.\n"
            "❌ Do NOT greet the user, do NOT explain the emoji, and do NOT provide extra text or formatting.\n"
            "\n"
            "Only return the translation in the specified one-line format — nothing more."
        ),


        model =model,
        tools=[
            turkish_agent.as_tool(
                tool_name="turkish_tool",
                tool_description="Handles Turkish translation questions and requests."
            ),
            chinese_agent.as_tool(
                tool_name="chinese_tool",
                tool_description="Handles Chinese translation questions and requests."
            ),
        
            arabic_agent.as_tool(
                tool_name="arabic_tool",
                tool_description="Handles arabic translation questions and requests."
        )
        ],
    )

    #result =await Runner.run(starting_agent=translator_agent ,input ="how to say egg in spanish", run_config =config)
    #print(result.final_output)
    result =Runner.run_streamed(starting_agent=translator_agent ,input ="tea is very cold in turkish", run_config =config)
    async for event in result.stream_events():
        if event.type =="raw_response_event" and isinstance(event.data,ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
  
   

if __name__ == "__main__":
    asyncio.run(main())



