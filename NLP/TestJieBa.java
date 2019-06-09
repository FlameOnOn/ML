package com.example.demo;

import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.WordDictionary;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

public class TestJieBa {
    public static void main(String args[]){
        System.out.println("testhaha") ;
        String test = "今天重庆的天气很好，朝天门大桥发生了一起追尾事故" ;
        JiebaSegmenter jiebaSegmenter = new JiebaSegmenter() ;
        System.out.println(jiebaSegmenter.sentenceProcess(test));


        Path path = Paths.get(new File( test.getClass().getClassLoader().getResource("dicts/jieba.dict").getPath() ).getAbsolutePath() ) ;
        WordDictionary.getInstance().loadUserDict(path);
        jiebaSegmenter = new JiebaSegmenter();
        System.out.println(jiebaSegmenter.sentenceProcess(test));

    }
}
