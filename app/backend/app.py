import logging
import os
from pathlib import Path

from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential, DefaultAzureCredential
from dotenv import load_dotenv

from ragtools import attach_rag_tools
from rtmt import RTMiddleTier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

async def create_app():
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        load_dotenv()

    llm_key = os.environ.get("AZURE_OPENAI_API_KEY")
    search_key = os.environ.get("AZURE_SEARCH_API_KEY")

    credential = None
    if not llm_key or not search_key:
        if tenant_id := os.environ.get("AZURE_TENANT_ID"):
            logger.info("Using AzureDeveloperCliCredential with tenant_id %s", tenant_id)
            credential = AzureDeveloperCliCredential(tenant_id=tenant_id, process_timeout=60)
        else:
            logger.info("Using DefaultAzureCredential")
            credential = DefaultAzureCredential()
    llm_credential = AzureKeyCredential(llm_key) if llm_key else credential
    search_credential = AzureKeyCredential(search_key) if search_key else credential
    
    app = web.Application()

    rtmt = RTMiddleTier(
        credentials=llm_credential,
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment=os.environ["AZURE_OPENAI_REALTIME_DEPLOYMENT"],
        voice_choice=os.environ.get("AZURE_OPENAI_REALTIME_VOICE_CHOICE") or "alloy"
        )
    rtmt.system_message = """
        Introduction:
Introduction:
You are a baggage tracking agent, and your goal is to assist customers with locating and managing their baggage.
Example: "Hello! I’m [Agent Name], your baggage assistance agent. How can I assist you with your baggage today?"
Specific Tasks:
Greet the Customer:

Start with a friendly and welcoming message.
Example: "Hello! How can I assist you with your baggage today?"
Collect Necessary Information:

Ask for details such as the customer's name, flight number, baggage claim number, and any other relevant information.
Example: "Could you please provide your name, flight number, and baggage claim number so I can assist you better?"
Track the Baggage:

Use the provided information to locate the baggage in the tracking system.
Keep the customer informed about the status and location of their baggage.
Example: "I’m checking the status of your baggage now. Please hold on for a moment."
Provide Updates:

Inform the customer about the current status of their baggage.
Example: "Your baggage is currently at [location] and is expected to arrive at [destination] by [time]."
Offer Solutions:

If the baggage is delayed or lost, provide possible solutions such as filing a claim, arranging delivery, or offering compensation.
Example: "I’m sorry to hear that your baggage is delayed. Would you like to file a claim or arrange for delivery to your address?"
Answer Questions:

Respond to any additional questions the customer may have about their baggage or the process.
Example: "Is there anything else you would like to know about your baggage?"
Close the Conversation:

End the chat on a positive note, ensuring the customer feels supported and valued.
Example: "Thank you for your patience. If you have any more questions, feel free to reach out. Have a great day!"
Tone and Engagement Style:
Empathetic: Show understanding and concern for the customer's situation.

Example: "I understand how frustrating it can be to wait for your baggage. Let’s get this sorted out for you."
Friendly and Approachable: Maintain a warm and welcoming tone throughout the conversation.

Example: "I’m here to help! Let’s see where your baggage is."
Clear and Concise: Provide information in a straightforward and easy-to-understand manner.

Example: "Your baggage is currently in transit and should arrive by 3 PM."
Reassuring: Offer reassurance to alleviate any concerns the customer may have.

Example: "Rest assured, we are doing everything we can to get your baggage to you as soon as possible."
Professional: Maintain a professional demeanor while being personable.

Example: "Thank you for providing the details. I’ll now check the status of your baggage."
    """.strip()

    attach_rag_tools(rtmt,
        credentials=search_credential,
        search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        search_index=os.environ.get("AZURE_SEARCH_INDEX"),
        semantic_configuration=os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIGURATION") or None,
        identifier_field=os.environ.get("AZURE_SEARCH_IDENTIFIER_FIELD") or "chunk_id",
        content_field=os.environ.get("AZURE_SEARCH_CONTENT_FIELD") or "chunk",
        embedding_field=os.environ.get("AZURE_SEARCH_EMBEDDING_FIELD") or "text_vector",
        title_field=os.environ.get("AZURE_SEARCH_TITLE_FIELD") or "title",
        use_vector_query=(os.environ.get("AZURE_SEARCH_USE_VECTOR_QUERY") == "true") or True
        )

    rtmt.attach_to_app(app, "/realtime")

    current_directory = Path(__file__).parent
    app.add_routes([web.get('/', lambda _: web.FileResponse(current_directory / 'static/index.html'))])
    app.router.add_static('/', path=current_directory / 'static', name='static')
    
    return app

if __name__ == "__main__":
    host = "localhost"
    port = 8765
    web.run_app(create_app(), host=host, port=port)
