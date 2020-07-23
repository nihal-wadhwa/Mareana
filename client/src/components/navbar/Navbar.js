import React from 'react';
import './navbar.style.scss'
import image from '../../images/mareana_logo.png'

const Navbar = () => (
    <nav class="navbar">
        <div class="container">
            <div class="logo">
                <img src={image} alt="logo" />
            </div>
            <ul class="nav">
                <a href="#">HOME</a>
            </ul>
        </div >
    </nav >
);

export default Navbar;