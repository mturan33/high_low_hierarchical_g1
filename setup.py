"""Setup for high_low_hierarchical_g1 package."""

from setuptools import setup, find_packages

setup(
    name="high_low_hierarchical_g1",
    version="0.1.0",
    description="Hierarchical LLM+RL Control for Unitree G1 Humanoid",
    author="Turan",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
    ],
    extras_require={
        "llm": ["anthropic", "openai"],
    },
)
