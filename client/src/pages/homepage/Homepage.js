import React from 'react';
import './homepage.style.scss';
import Navbar from '../../components/navbar/Navbar'
import Upload from '../../components/upload/Upload'
import upload from '../../images/upload.svg'
import process from '../../images/processing.svg'
import download from '../../images/folder.svg'

const Homepage = () => (
    <div className='homepage'>
        <Navbar></Navbar>
        <div className="intro">
            <div>
                <img src={upload} alt="upload" />
                <h4>Upload Your Documents</h4>
            </div>
            <div>
                <img src={process} alt="process" />
                <h4>We Label Them For You</h4>
            </div>
            <div>
                <img src={download} alt="download" />
                <h4>Results saved in your folder</h4>
            </div>
        </div>
        <Upload></Upload>
    </div >
);

export default Homepage;