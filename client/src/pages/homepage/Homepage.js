import React from 'react';
import './homepage.style.scss';
import Navbar from '../../components/navbar/Navbar'
import Upload from '../../components/upload/Upload'

const Homepage = () => (
    <div className='homepage'>
        <Navbar></Navbar>
        <h1 className='intro'>Document Symbol Classifier</h1>
        <Upload></Upload>
    </div >
);

export default Homepage;