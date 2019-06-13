from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.requests import Request
from fastai import *
from fastai.vision import *
import asyncio
import aiohttp
import uvicorn

classes = ['guitar', 'ukulele']
async def setup_learner():
    try:
        learn = load_learner('.','model2.pkl' )
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

app = Starlette (debug = True)

async def get_img_from_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()



def make_prediction(data):
    image = open_image(BytesIO(data))
    prediction = learn.predict(image)
    prediction_class = prediction[0]
    guitar_confidence = format(prediction[2][0].item(), '.6f')
    ukulele_confidence = format(prediction[2][1].item(), '.6f')
    print (guitar_confidence, ukulele_confidence)

    if float(guitar_confidence) < 0.9 and float(ukulele_confidence) < 0.9:
        prediction_class = 'Do not try to trick me asshole !'

    html_answer = """<h1>Voila the answer is: {}</h1><br>
                <h2>Guitar confidence: {} </h2><br>
                <h3>Ukulele confidence: {}</h3><br>
             """.format(prediction_class, guitar_confidence, ukulele_confidence)

    return HTMLResponse(html_answer)


@app.route('/')
async def homepage(request):
    return HTMLResponse("""
    <form action="/upload" method="post" enctype="multipart/form-data">
            Dawaj obrazek gytarry lub instrumentu
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Dawaj linka do zdjęcia:
        <form action="/dawaj-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Sssij i analizuj">
        </form>    
    """)



@app.route('/upload', methods = ['POST'])
async def upload(request):
    form = await request.form()
    data = await form["file"].read()
    return make_prediction(data)


@app.route('/dawaj-url', methods = ['GET'])
async def dawaj_url (request):
    url = request.query_params['url']
    data = await get_img_from_url(url)
    return make_prediction (data)





if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port = 8000)
