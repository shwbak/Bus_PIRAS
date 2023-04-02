using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using OpenCvSharp;
using System.Threading.Tasks;

public class MainTest : MonoBehaviour
{
    public Mat src;
    //src = Cv2.ImRead("dd.jpg", ImreadModes.Grayscale);
    private Texture2D texture;
    private Renderer rend;

    // Start is called before the first frame update
    void Start()
    {
        src = new Mat(new Size(775, 559), MatType.CV_8UC3);
        texture = new Texture2D(775, 559, TextureFormat.RGBA32, false);
    }

    // Update is called once per frame
    void Update()
    {
        //Cv2.Resize(src, src, new Size(640, 480));
        src = Cv2.ImRead("dd.jpg", ImreadModes.Grayscale);
        //MatToTexture(src, texture);

        rend = gameObject.GetComponent<Renderer>();
        if (rend != null)
        {
            rend.material.mainTexture = texture;
        }
    }

    public void MatToTexture(Mat sourceMat, Texture2D tex)
    {
        //Get the height and width of the Mat 
        int imgHeight = sourceMat.Height;
        int imgWidth = sourceMat.Width;

        byte[] matData = new byte[imgHeight * imgWidth];

        //Get the byte array and store in matData
        byte k = sourceMat.Get<byte>(0,0);
        matData[matData.Length-1] = k;
        //Create the Color array that will hold the pixels 
        Color32[] c = new Color32[imgHeight * imgWidth];

        //Get the pixel data from parallel loop
        Parallel.For(0, imgHeight, i => {
            for (var j = 0; j < imgWidth; j++)
            {
                byte vec = matData[j + i * imgWidth];
                var color32 = new Color32
                {
                    r = vec,
                    g = vec,
                    b = vec,
                    a = 0
                };
                c[j + i * imgWidth] = color32;
            }
        });

        //Create Texture from the result
        //Texture2D tex = new Texture2D(imgWidth, imgHeight, TextureFormat.RGBA32, true, true);
        tex.SetPixels32(c);
        tex.Apply();
    }
}
