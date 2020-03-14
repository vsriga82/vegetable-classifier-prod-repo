# Veg calssifier docker file

FROM python:3.7-slim-stretch

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip

RUN pip install numpy pandas sklearn jupyter fastai

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]  