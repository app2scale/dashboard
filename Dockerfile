FROM python:3.10
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
COPY --chown=user . $HOME/app
WORKDIR $HOME/app
RUN (cd agent & pip install -e .)

CMD ["solara", "run", "agent.dashboard",  "--host", "0.0.0.0", "--port", "7860"]
