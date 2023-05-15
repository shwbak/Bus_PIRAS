using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class MenuControll : MonoBehaviour
{
    public Button DB;
    public Button CCTV;

    public Sprite ClickImage;
    public Sprite NoneImage;
    
    public void Update()
    {
        
    }

    public void MenuButton(int type)
    {
        if (type == 1)
        {
            DB.GetComponent<Image>().sprite = ClickImage;
            CCTV.GetComponent<Image>().sprite = NoneImage;
        }
        else if (type == 2)
        {
            DB.GetComponent<Image>().sprite = NoneImage;
            CCTV.GetComponent<Image>().sprite = ClickImage;
        }
    }
}
