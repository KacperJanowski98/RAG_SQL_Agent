from langchain_ollama import ChatOllama
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from operator import itemgetter
from langchain_core.tools import tool
from typing_extensions import TypedDict
from typing_extensions import Annotated
from agent_graph.load_tools_config import LoadToolsConfig

TOOLS_CFG = LoadToolsConfig()


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


class ChinookSQLAgent:
    """
    A specialized SQL agent that interacts with the Chinook SQL database using an LLM (Large Language Model).

    The agent handles SQL queries by mapping user questions to relevant SQL tables based on categories like "Music"
    and "Business". It uses an extraction chain to determine relevant tables based on the question and then
    executes queries against the database using the appropriate tables.

    Attributes:
        sql_agent_llm (ChatOpenAI): The language model used for interpreting and interacting with the database.
        db (SQLDatabase): The SQL database object, representing the Chinook database.
        full_chain (Runnable): A chain of operations that maps user questions to SQL tables and executes queries.

    Methods:
        __init__: Initializes the agent by setting up the LLM, connecting to the SQL database, and creating query chains.

    Args:
        sqldb_directory (str): The directory where the Chinook SQLite database file is located.
        llm (str): The name of the LLM model to use (e.g., "gpt-3.5-turbo").
        llm_temperature (float): The temperature setting for the LLM, controlling the randomness of responses.
    """

    def __init__(self, sqldb_directory: str, llm: str, llm_temerature: float) -> None:
        """Initializes the ChinookSQLAgent with the LLM and database connection.

        Args:
            sqldb_directory (str): The directory path to the SQLite database file.
            llm (str): The LLM model identifier (e.g., "gpt-3.5-turbo").
            llm_temerature (float): The temperature value for the LLM, determining the randomness of the model's output.
        """
        self.sql_agent_llm = ChatOllama(
            model=llm, temperature=llm_temerature)
        self.system_role = """Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n
            Question: {question}\n
            SQL Query: {query}\n
            SQL Result: {result}\n
            Answer:
            """
        self.db = SQLDatabase.from_uri(
            f"sqlite:///{sqldb_directory}")
        print(self.db.get_usable_table_names())

        execute_query = QuerySQLDatabaseTool(db=self.db)
        answer_prompt = PromptTemplate.from_template(
            self.system_role)

        answer = answer_prompt | self.sql_agent_llm | StrOutputParser()
        self.full_chain = (
            RunnablePassthrough.assign(query=self.__write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer
        )

    def __write_query(self, state: State):
        query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
        prompt = query_prompt_template.invoke({
            "dialect": self.db.dialect,
            "top_k": 10,
            "table_info": self.db.get_table_info(),
            "input": state["question"],
        })
        structured_llm = self.sql_agent_llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return result

@tool
def query_chinook_sqldb(query: str) -> str:
    """Query the Chinook SQL Database. Input should be a search query."""
    # Create an instance of ChinookSQLAgent
    agent = ChinookSQLAgent(
        sqldb_directory=TOOLS_CFG.chinook_sqldb_directory,
        llm=TOOLS_CFG.chinook_sqlagent_llm,
        llm_temerature=TOOLS_CFG.chinook_sqlagent_llm_temperature
    )

    query = agent.full_chain.invoke({"question": query})

    return agent.db.run(query)
