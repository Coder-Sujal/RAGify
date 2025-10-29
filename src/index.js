import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import { OpenAIEmbeddings } from "@langchain/openai";
import OpenAI from "openai";

import dotenv from "dotenv";
import readline from "readline";
import path from "path";

dotenv.config();

//Load the pdf file

const pdfname = "nodejs_tutorial.pdf";
const filePath = path.join(process.cwd(), "content", pdfname);
const loader = new PDFLoader(filePath);
const docs = await loader.load();

//split the docs into chunk

const textSplitter = new CharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 400,
});
const chunks = await textSplitter.splitDocuments(docs);

//storing each chunk into vector db (here I am using qdrant db)

const client = new QdrantClient({ url: process.env.QDRANT_URL });

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large",
});

const vectorStore = await QdrantVectorStore.fromDocuments(chunks, embeddings, {
  client,
  collectionName: "pdf-chunk-collection",
});

//taking the prompt from user

async function readInputFromConsole(input_message) {
  const rl = readline.createInterface({
    input: process.stdin,
  });

  const ask = (query) => new Promise((resolve) => rl.question(query, resolve));
  console.log(`🤖 ${input_message} :- `);
  const prompt = await ask("");

  return prompt
}

//creating a system prompt

const SYSTEM_PROMPT = `
You are a helpful AI assistant.
Your task is to answer the user's query based solely on the context provided by the user.
Each piece of context will include the page number from which it was taken.
You must use only the given context to generate your answer and cite the corresponding page number(s) in your response. 
`;

const openaiClient = new OpenAI();

const messages = [];

messages.push({ content: SYSTEM_PROMPT, role: "system" });


while(true){
    //reading the input from user
    const prompt = await readInputFromConsole("Enter the prompt");
    
    //taking relavent chunks from the vectordb
    const retrievedChunks = await vectorStore.similaritySearch(prompt, 3);
    
    //creating a user prompt with chunks(for context)
    let context = "";
    
    retrievedChunks.map((retrievedChunk) => {
      context += `
        content :- ${retrievedChunk.pageContent}\n\n
        pageNumber :- ${retrievedChunk.metadata.loc.pageNumber}
        `;
    });
    
    const user_prompt = `
    user query :- ${prompt} \n\n
    context :- 
    ${context}
    `;
    
    //asking from llm
    messages.push({ content: user_prompt, role: "user" });

    const response = await openaiClient.chat.completions.create({
      model: "gpt-4.1-nano",
      messages: messages
    });
    
    //printing response
    console.log("🤖 : ", response.choices[0].message.content, "\n");

    messages.push({
        content: response.choices[0].message.content, role:"assistant"
    })
}