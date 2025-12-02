import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
my_api_key = os.getenv("GOOGLE_API_KEY")




def main():
    print("Hello from langchain-course!")
    information = """Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa's administrative capital.[1][2] He is of British and Pennsylvania Dutch ancestry.[3][4] His mother, Maye (née Haldeman), is a model and dietitian born in Saskatchewan, Canada, and raised in South Africa.[5][6][7] Musk therefore holds both South African and Canadian citizenship from birth.[8] His father, Errol Musk, is a South African electromechanical engineer, pilot, sailor, consultant, emerald dealer, and property developer, who partly owned a rental lodge at Timbavati Private Nature Reserve.[9][10][11][12]

His maternal grandfather, Joshua N. Haldeman, who died in a plane crash when Elon was a toddler, was an American-born Canadian chiropractor, aviator and political activist in the technocracy movement[13][14] who moved to South Africa in 1950.[15]

Elon has a younger brother, Kimbal, a younger sister, Tosca, and four paternal half-siblings.[16][17][7][18] Musk was baptized as a child in the Anglican Church of Southern Africa.[19][20] Despite both Elon and Errol previously stating that Errol was a part owner of a Zambian emerald mine,[12] in 2023, Errol recounted that the deal he made was to receive "a portion of the emeralds produced at three small mines".[21][22] Errol was elected to the Pretoria City Council as a representative of the anti-apartheid Progressive Party and has said that his children shared their father's dislike of apartheid.[1]

After his parents divorced in 1979, Elon, aged around 9, chose to live with his father because Errol Musk had an Encyclopædia Britannica and a computer.[23][3][9] Elon later regretted his decision and became estranged from his father.[24] Elon has recounted trips to a wilderness school that he described as a "paramilitary Lord of the Flies" where "bullying was a virtue" and children were encouraged to fight over rations.[25] In one incident, after an altercation with a fellow pupil, Elon was thrown down concrete steps and beaten severely, leading to him being hospitalized for his injuries.[26] Elon described his father berating him after he was discharged from the hospital.[26] Errol denied berating Elon and claimed, "The [other] boy had just lost his father to suicide, and Elon had called him stupid. Elon had a tendency to call people stupid. How could I possibly blame that child?"[27]"""

    summary_template = """
    given the information {information} about a person i want you to create:
    1. A short summary within 50 words
    2. Find two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatGoogleGenerativeAI(
        temperature=0,
        model="gemini-2.0-flash",
        api_key=my_api_key
    )

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)
    print(type(response))



if __name__ == "__main__":
    main()
