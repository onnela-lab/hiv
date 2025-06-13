import asyncio
import argparse
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from typing import Literal
import yaml


INSTRUCTIONS = """
You are an expert modeler with deep understanding of both sexual contact networks and
their implication for epidemiology and stochastic agent-based models. Based on your deep
experience, review the following manuscript and build a mental map of the manuscript.
Then generate a report on the manuscript paying careful attention to address all
requested points. Please follow the following guidance while compiling your report:

- Be conservative when compiling the report, e.g., use 'unknown' if you are unsure
instead of guessing.
"""


class Args:
    model: str
    document: Path
    output: Path | None


class Parameter(BaseModel):
    """Parameter of a pair-formation model."""

    symbol: str = Field(description="Mathematical symbol used in equations.")
    description: str = Field(
        description="Description of what the parameter represents."
    )
    estimate: float | Literal["unknown"] = Field(
        description="Best estimate of the parameter value based on the text of the "
        "manuscript. Do not try to estimate the parameter yourself. Only report what "
        "is presented in the manuscript."
    )
    location: str = Field(
        description="Where the parameter and estimate where found in the manuscript."
    )


class Summary(BaseModel):
    """Summary statistics about sexual contact networks."""

    description: str = Field(
        description="Description of the summary statistic including units."
    )
    value: float = Field(
        description="Value of the summary statistic reported in the manuscript."
    )
    location: str = Field(
        description="Where the summary and value where found in the manuscript."
    )


class Reference(BaseModel):
    """Reference that is likely to discuss or introduce pair formation models for sexual
    contact networks."""

    formatted: str = Field(description="Formatted reference.")
    doi: str | None = Field(description="DOI if available.")
    reason: str = Field(
        description="Reason why this reference is likely to discuss or introduce pair "
        "formation models."
    )


class Report(BaseModel):
    """Report on a pair formation model."""

    simulation_type: Literal["discrete", "continuous"] = Field(
        description="Indicates if the simulation is in continuous time (often "
        "presented as differential equations with rate parameters) or discrete time "
        "(often presented as step-wise simulations with probabilities of events "
        "occuring)."
    )
    network_type: Literal["unimodal", "bimodal", "other"] = Field(
        description="Type of the network such as 'unimodal' for homosexual contact "
        "networks, 'bimodal' for heterosexual contact networks, or some other "
        "structure.",
    )
    confidence: int = Field(
        ge=0,
        le=5,
        description="Rating from 0 to 5 of how confident you are about the information "
        "in the report, 0 indicating 'not confident at all' and 5 indicating "
        "'extremely confident'.",
    )
    infection_model: str | Literal["none"] = Field(
        description="Description of the infection model presented in the manuscript or "
        "'none' if there is none."
    )
    parameters: list[Parameter] = Field(
        description="Sequence of parameters used by the pair-formation model."
    )
    summaries: list[Summary] = Field(
        description="Sequence of summary statistics about sexual contact networks "
        "reported in the manuscript."
    )
    population_structure: str | Literal["unstructured"] = Field(
        description="Description of the structure of the population in the model, "
        "e.g., structure by age, education level, sexual activity level, etc., or "
        "'unstructured' if the population is homogeneous."
    )
    description: str = Field(
        description="Description of the pair formation model with enough details to "
        "reproduce the implementation. If the manuscript does not provide enough "
        "details, say so; do not try to fill in gaps."
    )
    references: list[Reference] = Field(
        description="Sequence of references that are likely to discuss or introduce "
        "pair formation models for sexual contact networks.",
    )
    title: str = Field(description="Title of the manuscript.")


class NoReport(BaseModel):
    """It is not possible to compile a report, e.g., if the manuscript does not present
    a pair-formation model."""

    reason: str = Field(
        description="Reason why it was not possible to create a report."
    )


async def __main__(argv: list[str] | None = None) -> None:
    load_dotenv()

    # Define and parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        help="AI model.",
        default="google-gla:gemini-2.5-pro-preview-05-06",
    )
    parser.add_argument("document", help="Input document.", type=Path)
    parser.add_argument("output", help="Output document.", type=Path, nargs="?")
    args: Args = parser.parse_args(argv)  # type: ignore

    # Build the agent.
    agent = Agent(
        model=args.model,
        output_type=Report | NoReport,  # type: ignore
        instructions=INSTRUCTIONS.strip(),
    )

    # Load and process the document.
    assert args.document.suffix == ".pdf"
    document_bytes = args.document.read_bytes()
    result = await agent.run(
        [BinaryContent(document_bytes, media_type="application/pdf")]
    )
    output: Report | NoReport = result.output  # type: ignore

    # Check a report was generated.
    if isinstance(output, NoReport):
        raise ValueError(output.reason)

    # Write a json-representation to disk and the terminal.
    yaml_text = yaml.dump(output.model_dump(), allow_unicode=True)
    print(yaml_text)
    output_path = args.output
    if not output_path:
        output_path = args.document.with_suffix(".yaml")
    output_path.write_text(yaml_text)


if __name__ == "__main__":
    asyncio.run(__main__())
